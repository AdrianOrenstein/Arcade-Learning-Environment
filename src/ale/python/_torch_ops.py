"""PyTorch custom ops for ALE - registered via AtariVectorEnv.torch().

No 'from __future__ import annotations' here so torch.library.custom_op can
infer op schemas from type annotations without PEP 563 string-ification.

Do not import this module at the top level; it is lazy-loaded by torch().
"""

import numpy as np
import torch
from torch.library import EffectType

__all__ = ["register_pytorch_ops"]

_torch_registered: bool = False
_torch_buffers: dict[int, dict] = {}

_INFO_KEYS = ("env_id", "lives", "frame_number", "episode_frame_number")
_ALL_KEYS = ("obs", "reward", "term", "trunc") + _INFO_KEYS


def register_pytorch_ops(env, device=None, tensordict: bool = False):
    """Allocate pinned buffers and register ale::* torch custom ops once.

    Patches env.step / env.send / env.recv to return tensors and returns env.
    """
    global _torch_registered

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

    _torch_buffers[handle_id] = {
        "device": device,
        "vector_interface": env.ale,
        "actions": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "obs": torch.empty(obs_shape, dtype=torch.uint8, pin_memory=True),
        "reward": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "term": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "trunc": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "paddle_strength": np.ones(num_envs, dtype=np.float32),
        "h2d_event": torch.cuda.Event() if torch.cuda.is_available() else None,
        **{
            k: torch.empty(num_envs, dtype=torch.int32, pin_memory=True)
            for k in _INFO_KEYS
        },
    }

    if not _torch_registered:
        _torch_registered = True

        @torch.library.custom_op("ale::send", mutates_args=())
        def ale_send(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            buf["actions"].copy_(actions)
            buf["vector_interface"].send(buf["actions"].numpy(), buf["paddle_strength"])
            return actions.new_empty(())

        ale_send.register_effect(EffectType.ORDERED)

        @ale_send.register_fake
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty(())

        @torch.library.custom_op("ale::recv", mutates_args=())
        def ale_recv(
            handle_id: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            buf = _torch_buffers[handle_id]
            # Ensure the previous H2D copy has finished before overwriting the
            # pinned CPU buffers it was reading from.
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

            dev = buf["device"]
            all_bufs = [buf["obs"], buf["reward"], buf["term"], buf["trunc"]] + [
                buf[k] for k in _INFO_KEYS
            ]
            if dev is not None:
                out = tuple(t.to(dev, non_blocking=True) for t in all_bufs)
                buf["h2d_event"].record()
                return out
            return tuple(t.clone() for t in all_bufs)

        ale_recv.register_effect(EffectType.ORDERED)

        @ale_recv.register_fake
        def _(
            handle_id: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            buf = _torch_buffers[handle_id]
            dev = buf["device"]
            dtypes = [
                buf["obs"].dtype,
                buf["reward"].dtype,
                buf["term"].dtype,
                buf["trunc"].dtype,
            ]
            dtypes += [buf[k].dtype for k in _INFO_KEYS]
            shapes = [
                buf["obs"].shape,
                buf["reward"].shape,
                buf["term"].shape,
                buf["trunc"].shape,
            ]
            shapes += [buf[k].shape for k in _INFO_KEYS]
            return tuple(
                torch.empty(s, dtype=d, device=dev) for s, d in zip(shapes, dtypes)
            )

    def unregister() -> None:
        _torch_buffers.pop(handle_id, None)

    if tensordict:

        def step(actions: torch.Tensor):
            torch.ops.ale.send(handle_id, actions)
            return _TensorDict(
                dict(zip(_ALL_KEYS, torch.ops.ale.recv(handle_id))),
                batch_size=[num_envs],
            )

        def recv():
            return _TensorDict(
                dict(zip(_ALL_KEYS, torch.ops.ale.recv(handle_id))),
                batch_size=[num_envs],
            )

    else:

        def step(actions: torch.Tensor):
            torch.ops.ale.send(handle_id, actions)
            return torch.ops.ale.recv(handle_id)

        def recv():
            return torch.ops.ale.recv(handle_id)

    def send(actions: torch.Tensor) -> None:
        torch.ops.ale.send(handle_id, actions)

    return step, send, recv, unregister
