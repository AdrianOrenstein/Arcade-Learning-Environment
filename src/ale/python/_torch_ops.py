"""PyTorch custom ops for ALE - registered via AtariVectorEnv.torch().

No 'from __future__ import annotations' here so torch.library.custom_op can
infer op schemas from type annotations without PEP 563 string-ification.

Two backends share the same step/send/recv interface:
  * cpp=False (default): pure-Python ale::send / ale::recv custom ops.
  * cpp=True: ale_cpp::send / ale_cpp::recv implemented in C++ (needs ale_py
    built with -DBUILD_VECTOR_TORCH_LIB=ON).

Do not import this module at the top level; it is lazy-loaded by torch().
"""

import numpy as np
import torch
from torch.library import EffectType

__all__ = ["register_pytorch_ops"]

_torch_registered: bool = False
_torch_cpp_registered: bool = False
_torch_buffers: dict[int, dict] = {}
# Keeps torch.compile effect-registration handles alive for the process lifetime.
_effect_handles: list = []

_INFO_KEYS = ("env_id", "lives", "frame_number", "episode_frame_number")
_ALL_KEYS = ("obs", "reward", "term", "trunc") + _INFO_KEYS


def _is_same_step(env) -> bool:
    """True if the env autoresets in the same step (final_obs is returned)."""
    mode = getattr(env, "autoreset_mode", None)
    return getattr(mode, "name", None) == "SAME_STEP"


def _python_recv_impl(handle_id: int, with_final: bool):
    """Shared body for the ale::recv / ale::recv_same_step custom ops."""
    buf = _torch_buffers[handle_id]
    # Ensure the previous H2D copy finished before overwriting the pinned buffers.
    if buf["h2d_event"] is not None:
        buf["h2d_event"].synchronize()

    obs_np, reward_np, term_np, trunc_np, info = buf["vector_interface"].recv()
    for t, src in zip(
        [buf["obs"], buf["reward"], buf["term"], buf["trunc"]],
        [obs_np, reward_np, term_np, trunc_np],
    ):
        t.copy_(torch.from_numpy(src))
    for k in _INFO_KEYS:
        buf[k].copy_(torch.from_numpy(info[k]))

    all_bufs = [buf["obs"], buf["reward"], buf["term"], buf["trunc"]]
    all_bufs += [buf[k] for k in _INFO_KEYS]
    if with_final:
        # final_obs is only populated by ALE for done envs (and absent when none
        # are done); zero the rest so non-done entries are well-defined.
        if "final_obs" in info:
            buf["final_obs"].copy_(torch.from_numpy(info["final_obs"]))
        else:
            buf["final_obs"].zero_()
        all_bufs.append(buf["final_obs"])

    dev = buf["device"]
    if dev is not None:
        out = tuple(t.to(dev, non_blocking=True) for t in all_bufs)
        if buf["h2d_event"] is not None:
            buf["h2d_event"].record()
        return out
    return tuple(t.clone() for t in all_bufs)


def _python_recv_fake(handle_id: int, with_final: bool):
    buf = _torch_buffers[handle_id]
    dev = buf["device"]
    keys = ["obs", "reward", "term", "trunc", *_INFO_KEYS]
    if with_final:
        keys.append("final_obs")
    return tuple(
        torch.empty(buf[k].shape, dtype=buf[k].dtype, device=dev) for k in keys
    )


def register_pytorch_ops(env, device=None, tensordict: bool = False, cpp: bool = False):
    """Register ale[_cpp]::send/recv ops and build step/send/recv/reset closures.

    Returns ``(step, send, recv, reset, unregister)``. ``step``/``recv``/``reset``
    return a TensorDict when ``tensordict=True``, else flat tuples; ``step`` and
    ``recv`` yield 8 tensors (9 with a trailing ``final_obs`` in SameStep mode).
    """
    _TensorDict = None
    if tensordict:
        try:
            from tensordict import TensorDict as _TensorDict
        except ImportError as e:
            raise ImportError(
                "tensordict=True requires the tensordict package. "
                "Install with: pip install tensordict"
            ) from e

    handle_id = id(env)
    num_envs = env.num_envs
    assert env.single_observation_space.shape is not None
    obs_shape = (num_envs,) + env.single_observation_space.shape
    # Whether step results should land on a CUDA device.
    targets_cuda = device is not None and torch.device(device).type == "cuda"

    setup = _setup_cpp if cpp else _setup_python
    send_op, recv_op, all_keys, unregister = setup(
        env, device, handle_id, num_envs, obs_shape, targets_cuda
    )

    if tensordict:

        def step(actions: torch.Tensor):
            send_op(handle_id, actions)
            return _TensorDict(
                dict(zip(all_keys, recv_op(handle_id))), batch_size=[num_envs]
            )

        def recv():
            return _TensorDict(
                dict(zip(all_keys, recv_op(handle_id))), batch_size=[num_envs]
            )

    else:

        def step(actions: torch.Tensor):
            send_op(handle_id, actions)
            return recv_op(handle_id)

        def recv():
            return recv_op(handle_id)

    def send(actions: torch.Tensor) -> None:
        send_op(handle_id, actions)

    def reset(*, seed=None, options=None):
        # reset() isn't on the hot path, so just convert the numpy result to
        # tensors on the target device (call the class method to avoid the patch).
        obs, info = type(env).reset(env, seed=seed, options=options)
        obs_t = torch.from_numpy(obs)
        info_t = {k: torch.from_numpy(v) for k, v in info.items()}
        if device is not None:
            obs_t = obs_t.to(device)
            info_t = {k: v.to(device) for k, v in info_t.items()}
        if tensordict:
            return _TensorDict(
                {"obs": obs_t, **info_t}, batch_size=[num_envs], device=device
            )
        return obs_t, info_t

    return step, send, recv, reset, unregister


def _setup_cpp(env, device, handle_id, num_envs, obs_shape, targets_cuda):
    """Register an env with the C++ torch backend (torch.ops.ale_cpp)."""
    global _torch_cpp_registered
    from ale_py import _ale_py as _ale_cpp

    if not hasattr(_ale_cpp, "torch_register_handle"):
        raise RuntimeError(
            "torch(cpp=True) requires ale_py built with the C++ torch backend; "
            "rebuild with -DBUILD_VECTOR_TORCH_LIB=ON."
        )
    if targets_cuda and not _ale_cpp.VECTOR_TORCH_CUDA:
        raise RuntimeError(
            "torch(cpp=True, device='cuda') requires ale_py built against a CUDA "
            "PyTorch (-DBUILD_VECTOR_TORCH_CUDA)."
        )

    # Persistent pinned D2H buffer (CUDA send target); C++ holds the raw pointer.
    # Discrete: int32[num_envs] action ids. Continuous: float32[num_envs*3] polar
    # actions, mapped to ids + paddle strength in C++ via map_action_idx.
    if env.continuous:
        pinned_in = torch.empty(
            num_envs * 3, dtype=torch.float32, pin_memory=targets_cuda
        )
        map_action_idx = (
            torch.from_numpy(env.map_action_idx).to(torch.int32).contiguous()
        )
        threshold = float(env.continuous_action_threshold)
        map_ptr = map_action_idx.data_ptr()
    else:
        pinned_in = torch.empty(num_envs, dtype=torch.int32, pin_memory=targets_cuda)
        map_action_idx = None
        threshold = 0.0
        map_ptr = 0
    _torch_buffers[handle_id] = {
        "cpp": True,
        "device": device,
        "num_envs": num_envs,
        "obs_shape": obs_shape,
        "actions": pinned_in,
        "map_action_idx": map_action_idx,  # keep alive (C++ copied it at register)
    }

    if not _torch_cpp_registered:
        _torch_cpp_registered = True

        def _recv_fake(handle_id: int, *, final: bool):
            buf = _torch_buffers[handle_id]
            dev = buf["device"]
            n = buf["num_envs"]
            obs = torch.empty(buf["obs_shape"], dtype=torch.uint8, device=dev)
            scalars = [
                torch.empty(n, dtype=d, device=dev)
                for d in (
                    torch.int32,
                    torch.bool,
                    torch.bool,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                    torch.int32,
                )
            ]
            out = (obs, *scalars)
            if final:
                out += (torch.empty(buf["obs_shape"], dtype=torch.uint8, device=dev),)
            return out

        @torch.library.register_fake("ale_cpp::send")
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty((0,))

        @torch.library.register_fake("ale_cpp::recv")
        def _(handle_id: int):
            return _recv_fake(handle_id, final=False)

        @torch.library.register_fake("ale_cpp::recv_same_step")
        def _(handle_id: int):
            return _recv_fake(handle_id, final=True)

        # Order send before recv under torch.compile (the ops have side effects
        # the schema can't express). Handles are kept alive in _effect_handles.
        from torch._higher_order_ops.effects import _register_effectful_op

        for _name in ("ale_cpp::send", "ale_cpp::recv", "ale_cpp::recv_same_step"):
            _effect_handles.append(_register_effectful_op(_name, EffectType.ORDERED))

    _ale_cpp.torch_register_handle(
        handle_id,
        env.ale,
        pinned_in.data_ptr(),
        num_envs,
        targets_cuda,
        env.continuous,
        threshold,
        map_ptr,
    )

    def unregister() -> None:
        # Drop the C++ handle first so no op can reference the freed pinned buffer.
        _ale_cpp.torch_unregister_handle(handle_id)
        _torch_buffers.pop(handle_id, None)

    if _is_same_step(env):
        return (
            torch.ops.ale_cpp.send,
            torch.ops.ale_cpp.recv_same_step,
            _ALL_KEYS + ("final_obs",),
            unregister,
        )
    return torch.ops.ale_cpp.send, torch.ops.ale_cpp.recv, _ALL_KEYS, unregister


def _setup_python(env, device, handle_id, num_envs, obs_shape, targets_cuda):
    """Register the pure-Python torch backend (torch.ops.ale)."""
    global _torch_registered

    same_step = _is_same_step(env)
    _torch_buffers[handle_id] = {
        "device": device,
        "vector_interface": env.ale,
        "continuous": env.continuous,
        "continuous_action_threshold": env.continuous_action_threshold,
        "map_action_idx": (
            torch.from_numpy(env.map_action_idx) if env.continuous else None
        ),
        "actions": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "paddle_strength": np.ones(num_envs, dtype=np.float32),
        "obs": torch.empty(obs_shape, dtype=torch.uint8, pin_memory=True),
        "reward": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "term": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "trunc": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        # The H2D-completion event is only meaningful when copying into a CUDA
        # device; the CPU path neither records nor waits on it, so don't allocate
        # one there (it would needlessly init a CUDA context on CPU-only runs).
        "h2d_event": torch.cuda.Event() if targets_cuda else None,
        **{
            k: torch.empty(num_envs, dtype=torch.int32, pin_memory=True)
            for k in _INFO_KEYS
        },
        # SameStep mode returns the pre-reset observation for done envs.
        **(
            {"final_obs": torch.empty(obs_shape, dtype=torch.uint8, pin_memory=True)}
            if same_step
            else {}
        ),
    }

    if not _torch_registered:
        _torch_registered = True

        @torch.library.custom_op("ale::send", mutates_args=())
        def ale_send(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            if buf["continuous"]:
                x = actions[:, 0] * torch.cos(actions[:, 1])
                y = actions[:, 0] * torch.sin(actions[:, 1])
                t = buf["continuous_action_threshold"]
                horizontal = -(x < -t).int() + (x > t).int() + 1
                vertical = -(y < -t).int() + (y > t).int() + 1
                fire = (actions[:, 2] > t).int()
                buf["actions"].copy_(buf["map_action_idx"][horizontal, vertical, fire])
                paddle = actions[:, 0]
                if paddle.is_cuda:
                    paddle = paddle.cpu()
                buf["paddle_strength"][:] = paddle.numpy()
            else:
                buf["actions"].copy_(actions)
            buf["vector_interface"].send(buf["actions"].numpy(), buf["paddle_strength"])
            return actions.new_empty(())

        ale_send.register_effect(EffectType.ORDERED)

        @ale_send.register_fake
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty(())

        _Recv8 = tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        _Recv9 = tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]

        @torch.library.custom_op("ale::recv", mutates_args=())
        def ale_recv(handle_id: int) -> _Recv8:
            return _python_recv_impl(handle_id, with_final=False)

        ale_recv.register_effect(EffectType.ORDERED)
        ale_recv.register_fake(lambda handle_id: _python_recv_fake(handle_id, False))

        @torch.library.custom_op("ale::recv_same_step", mutates_args=())
        def ale_recv_same_step(handle_id: int) -> _Recv9:
            return _python_recv_impl(handle_id, with_final=True)

        ale_recv_same_step.register_effect(EffectType.ORDERED)
        ale_recv_same_step.register_fake(
            lambda handle_id: _python_recv_fake(handle_id, True)
        )

    def unregister() -> None:
        _torch_buffers.pop(handle_id, None)

    if same_step:
        return (
            torch.ops.ale.send,
            torch.ops.ale.recv_same_step,
            _ALL_KEYS + ("final_obs",),
            unregister,
        )
    return torch.ops.ale.send, torch.ops.ale.recv, _ALL_KEYS, unregister
