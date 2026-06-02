"""Tests for PyTorch custom ops (ale::send / ale::recv / ale::step)."""

import numpy as np
import pytest
from ale_py import AtariVectorEnv

torch = pytest.importorskip("torch")

from ale_py._torch_ops import _torch_buffers  # noqa: E402

_INFO_KEYS = ("env_id", "lives", "frame_number", "episode_frame_number")
_GAME = "pong"
_NUM_ENVS = 2


def _send_recv(env, device=None):
    actions = torch.zeros(_NUM_ENVS, dtype=torch.int32)
    if device:
        actions = actions.to(device)
    torch.ops.ale.send(id(env), actions)
    return torch.ops.ale.recv(id(env))


def test_buffers_empty_after_close():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    assert id(env) in _torch_buffers
    env.close()
    assert id(env) not in _torch_buffers


def test_no_event_in_buffers():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    events = [
        k for k, v in _torch_buffers[id(env)].items() if isinstance(v, torch.cuda.Event)
    ]
    env.close()
    assert events == [], f"torch.cuda.Event found in buffers: {events}"


def test_recv_returns_8_tensors():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    result = _send_recv(env)
    env.close()
    assert len(result) == 8, f"expected 8 tensors, got {len(result)}"


def test_recv_dtypes():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    obs, reward, term, trunc, env_id, lives, frame_num, ep_frame_num = _send_recv(env)
    env.close()
    assert obs.dtype == torch.uint8
    assert reward.dtype == torch.int32
    assert term.dtype == torch.bool
    assert trunc.dtype == torch.bool
    assert env_id.dtype == torch.int32
    assert lives.dtype == torch.int32
    assert frame_num.dtype == torch.int32
    assert ep_frame_num.dtype == torch.int32


def test_no_aliasing():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    obs_0, *_ = _send_recv(env)
    obs_0_copy = obs_0.clone()
    _send_recv(env)
    env.close()
    assert torch.equal(obs_0, obs_0_copy), "obs from call N was overwritten by call N+1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_placement():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(device="cuda")
    result = _send_recv(env, device="cuda")
    env.close()
    for i, t in enumerate(result):
        assert t.is_cuda, f"tensor {i} is not on CUDA"


# def test_step_matches_numpy():
#     env_np = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
#     env_t = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
#     env_np.reset(seed=0)
#     env_t.reset(seed=0)
#     env_t.torch()

#     rng = np.random.default_rng(1)
#     for _ in range(20):
#         acts = rng.integers(0, 16, size=_NUM_ENVS, dtype=np.int32)
#         obs_np, rew_np, term_np, trunc_np, info_np = env_np.step(acts)

#         torch.ops.ale.send(id(env_t), torch.from_numpy(acts))
#         obs_t, rew_t, term_t, trunc_t, *info_tensors = torch.ops.ale.recv(id(env_t))

#         assert np.array_equal(obs_np, obs_t.numpy())
#         assert np.array_equal(rew_np, rew_t.numpy())
#         assert np.array_equal(term_np, term_t.numpy())
#         assert np.array_equal(trunc_np, trunc_t.numpy())
#         for key, tensor in zip(_INFO_KEYS, info_tensors):
#             assert np.array_equal(info_np[key], tensor.numpy()), f"mismatch on {key}"

#     env_np.close()
#     env_t.close()


def test_patched_step():
    """After .torch(), env.step/send/recv accept tensors and return 8-tuples."""
    env_np = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_t = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_np.reset(seed=0)
    env_t.reset(seed=0)
    env_t.torch()

    rng = np.random.default_rng(2)
    for _ in range(3):
        acts = rng.integers(0, 16, size=_NUM_ENVS, dtype=np.int32)
        obs_np, rew_np, term_np, trunc_np, info_np = env_np.step(acts)
        obs_t, rew_t, term_t, trunc_t, *info_tensors = env_t.step(
            torch.from_numpy(acts)
        )

        assert np.array_equal(obs_np, obs_t.numpy())
        assert np.array_equal(rew_np, rew_t.numpy())
        assert np.array_equal(term_np, term_t.numpy())
        assert np.array_equal(trunc_np, trunc_t.numpy())
        for key, tensor in zip(_INFO_KEYS, info_tensors):
            assert np.array_equal(info_np[key], tensor.numpy()), f"mismatch on {key}"

    env_np.close()
    env_t.close()


def test_torch_returns_self():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    result = env.torch()
    env.close()
    assert result is env
    assert callable(env.send)
    assert callable(env.step)
    assert callable(env.recv)


def test_compile():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch()
    compiled = torch.compile(env.step, fullgraph=True, mode="reduce-overhead")
    actions = torch.zeros(_NUM_ENVS, dtype=torch.int32)
    result = compiled(actions)
    env.close()
    assert len(result) == 8
