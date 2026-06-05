"""Tests for the TorchRL wrapper (AtariVectorEnv.torchrl())."""

import numpy as np
import pytest
from ale_py import AtariVectorEnv

torch = pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("torchrl")

from tensordict import TensorDict  # noqa: E402
from torchrl.collectors import SyncDataCollector  # noqa: E402
from torchrl.envs.utils import check_env_specs  # noqa: E402

_GAME = "pong"
_NUM_ENVS = 2
_PONG_ACTIONS = 6  # minimal action set size for Pong


def test_torchrl_specs():
    """torchrl's built-in spec validator accepts the wrapped env."""
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    tr = env.torchrl()
    try:
        check_env_specs(tr)
    finally:
        tr.close()


def test_torchrl_specs_continuous():
    """Spec validation also holds for the continuous action space."""
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, continuous=True)
    tr = env.torchrl()
    try:
        assert tr.action_spec.shape == torch.Size([_NUM_ENVS, 3])
        assert tr.action_spec.dtype == torch.float32
        check_env_specs(tr)
    finally:
        tr.close()


def test_torchrl_step_matches_numpy():
    """A TorchRL rollout reproduces the plain numpy step() rollout frame-for-frame."""
    env_np = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_np.reset(seed=0)

    tr = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS).torchrl()
    tr.set_seed(0)
    tr.reset()

    rng = np.random.default_rng(0)
    for _ in range(10):
        acts = rng.integers(0, _PONG_ACTIONS, size=_NUM_ENVS, dtype=np.int64)
        obs_np, rew_np, term_np, trunc_np, _ = env_np.step(acts)

        td = TensorDict({"action": torch.from_numpy(acts)}, batch_size=[_NUM_ENVS])
        nxt = tr.step(td)["next"]

        assert np.array_equal(nxt["obs"].cpu().numpy(), obs_np)
        assert np.array_equal(
            nxt["reward"].cpu().numpy().ravel(), rew_np.astype(np.float32)
        )
        assert np.array_equal(nxt["terminated"].cpu().numpy().ravel(), term_np)
        assert np.array_equal(nxt["truncated"].cpu().numpy().ravel(), trunc_np)

    env_np.close()
    tr.close()


def test_torchrl_async_split():
    """async_step_send / async_step_recv step the env in two halves."""
    tr = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS).torchrl()
    tr.set_seed(0)
    td = tr.reset()

    action_td = tr.rand_action(td)
    tr.async_step_send(action_td)
    nxt = tr.async_step_recv()

    assert nxt["obs"].shape == torch.Size([_NUM_ENVS, 4, 84, 84])
    assert nxt["obs"].dtype == torch.uint8
    assert nxt["done"].shape == torch.Size([_NUM_ENVS, 1])
    tr.close()


def test_torchrl_collector():
    """A SyncDataCollector drives the env and yields the requested frame count."""
    tr = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS).torchrl()
    n_frames = _NUM_ENVS * 5  # 10 frames
    collector = SyncDataCollector(
        tr,
        policy=None,
        frames_per_batch=n_frames,
        total_frames=n_frames,
    )
    try:
        batch = next(iter(collector))
    finally:
        collector.shutdown()

    assert batch.numel() == n_frames
    assert batch["next", "obs"].shape == torch.Size([_NUM_ENVS, 5, 4, 84, 84])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torchrl_cuda_device():
    """When device='cuda', specs and rollout tensors live on the GPU."""
    tr = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS).torchrl(device="cuda")
    try:
        check_env_specs(tr)
        td = TensorDict(
            {"action": torch.zeros(_NUM_ENVS, dtype=torch.int64, device="cuda")},
            batch_size=[_NUM_ENVS],
        )
        tr.set_seed(0)
        tr.reset()
        nxt = tr.step(td)["next"]
        assert nxt["obs"].is_cuda
        assert nxt["reward"].is_cuda
    finally:
        tr.close()
