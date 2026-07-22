"""Tests for PyTorch custom ops (ale::send / ale::recv / ale::step)."""

import ale_py
import numpy as np
import pytest
from ale_py import AtariVectorEnv

torch = pytest.importorskip("torch")

from ale_py._torch_ops import _torch_buffers  # noqa: E402

_INFO_KEYS = ("env_id", "lives", "frame_number", "episode_frame_number")
_GAME = "pong"
_NUM_ENVS = 2

# The C++ torch backend (torch(cpp=True)) is an opt-in build feature.
_CPP_AVAILABLE = hasattr(ale_py._ale_py, "torch_register_handle")
_CPP_CUDA = _CPP_AVAILABLE and ale_py._ale_py.VECTOR_TORCH_CUDA
cpp_required = pytest.mark.skipif(
    not _CPP_AVAILABLE, reason="ale_py built without the C++ torch backend"
)


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
    assert reward.dtype == torch.float32
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

    n_actions = int(env_np.action_space.nvec[0])
    rng = np.random.default_rng(2)
    for _ in range(3):
        acts = rng.integers(0, n_actions, size=_NUM_ENVS, dtype=np.int32)
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


# --- C++ torch backend (torch(cpp=True)) -----------------------------------


@cpp_required
def test_cpp_backend_active():
    """torch(cpp=True) registers a C++-backed handle."""
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(cpp=True)
    assert _torch_buffers[id(env)]["cpp"] is True
    env.close()
    assert id(env) not in _torch_buffers


@cpp_required
def test_cpp_recv_dtypes():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(cpp=True)
    obs, reward, term, trunc, env_id, lives, frame_num, ep_frame_num = env.step(
        torch.zeros(_NUM_ENVS, dtype=torch.int32)
    )
    env.close()
    assert obs.shape == (_NUM_ENVS, 4, 84, 84) and obs.dtype == torch.uint8
    assert reward.dtype == torch.float32
    assert term.dtype == torch.bool and trunc.dtype == torch.bool
    for t in (env_id, lives, frame_num, ep_frame_num):
        assert t.dtype == torch.int32


@cpp_required
def test_cpp_matches_python():
    """The C++ backend reproduces the Python backend frame-for-frame."""
    env_py = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_cpp = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_py.reset(seed=0)
    env_cpp.reset(seed=0)
    env_py.torch(cpp=False)
    env_cpp.torch(cpp=True)

    rng = np.random.default_rng(0)
    for _ in range(8):
        acts = torch.from_numpy(rng.integers(0, 6, size=_NUM_ENVS, dtype=np.int32))
        out_py = env_py.step(acts)
        out_cpp = env_cpp.step(acts)
        for a, b in zip(out_py, out_cpp):
            assert torch.equal(a, b)

    env_py.close()
    env_cpp.close()


@cpp_required
def test_cpp_no_aliasing():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(cpp=True)
    obs_0 = env.step(torch.zeros(_NUM_ENVS, dtype=torch.int32))[0]
    obs_0_copy = obs_0.clone()
    env.step(torch.zeros(_NUM_ENVS, dtype=torch.int32))
    env.close()
    assert torch.equal(obs_0, obs_0_copy), "obs from call N was overwritten by call N+1"


@cpp_required
def test_cpp_continuous_matches_python():
    """The C++ continuous mapping reproduces the Python one frame-for-frame."""
    env_py = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, continuous=True)
    env_cpp = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, continuous=True)
    env_py.reset(seed=0)
    env_cpp.reset(seed=0)
    env_py.torch(cpp=False)
    env_cpp.torch(cpp=True)

    # Clean polar actions (radius 1, axis-aligned angles) that sit well away from
    # the 0.5 threshold, so both backends discretize to the same action id.
    thetas = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    for k in range(8):
        act = torch.tensor(
            [[1.0, thetas[k % 4], float(k % 2)]] * _NUM_ENVS, dtype=torch.float32
        )
        for a, b in zip(env_py.step(act), env_cpp.step(act)):
            assert torch.equal(a, b)

    env_py.close()
    env_cpp.close()


@pytest.mark.skipif(not _CPP_CUDA, reason="no CUDA C++ torch backend")
def test_cpp_cuda_continuous():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, continuous=True)
    env.reset(seed=0)
    env.torch(device="cuda", cpp=True)
    act = torch.tensor(
        [[1.0, 0.0, 0.0]] * _NUM_ENVS, dtype=torch.float32, device="cuda"
    )
    result = env.step(act)
    env.close()
    assert result[0].is_cuda and result[0].shape == (_NUM_ENVS, 4, 84, 84)


@pytest.mark.skipif(not _CPP_CUDA, reason="no CUDA C++ torch backend")
def test_cpp_cuda_placement():
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(device="cuda", cpp=True)
    result = env.step(torch.zeros(_NUM_ENVS, dtype=torch.int32, device="cuda"))
    env.close()
    for i, t in enumerate(result):
        assert t.is_cuda, f"tensor {i} is not on CUDA"


# --- reset / final_obs / compile (both backends) ---------------------------


@pytest.mark.parametrize("cpp", [False, pytest.param(True, marks=cpp_required)])
def test_reset_returns_tensors(cpp):
    """After .torch(), reset() returns tensors (and a TensorDict in td mode)."""
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(cpp=cpp)
    obs, info = env.reset(seed=0)
    env.close()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (_NUM_ENVS, 4, 84, 84) and obs.dtype == torch.uint8
    for k in _INFO_KEYS:
        assert isinstance(info[k], torch.Tensor) and info[k].dtype == torch.int32


@pytest.mark.parametrize("cpp", [False, pytest.param(True, marks=cpp_required)])
def test_same_step_final_obs(cpp):
    """SameStep mode returns a 9th tensor (final_obs) matching the numpy path."""
    from gymnasium.vector import AutoresetMode

    kw = dict(
        num_envs=_NUM_ENVS,
        autoreset_mode=AutoresetMode.SAME_STEP,
        max_num_frames_per_episode=20,  # short episodes -> a truncation happens fast
    )
    env_np = AtariVectorEnv(_GAME, **kw)
    env_np.reset(seed=0)
    env_t = AtariVectorEnv(_GAME, **kw)
    env_t.reset(seed=0)
    env_t.torch(cpp=cpp)

    saw_done = False
    for _ in range(20):
        acts = np.zeros(_NUM_ENVS, dtype=np.int32)
        _, _, term_np, trunc_np, info_np = env_np.step(acts)
        out = env_t.step(torch.from_numpy(acts))
        assert len(out) == 9
        done = term_np | trunc_np
        if done.any() and "final_obs" in info_np:
            saw_done = True
            assert np.array_equal(
                out[8][done].cpu().numpy(), info_np["final_obs"][done]
            )

    env_np.close()
    env_t.close()
    assert saw_done, "expected a truncation to exercise final_obs"


@cpp_required
def test_cpp_compile():
    """The C++ ops are torch.compile-friendly (ORDERED effects registered)."""
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env.reset(seed=0)
    env.torch(cpp=True)
    compiled = torch.compile(env.step, fullgraph=True, mode="reduce-overhead")
    result = compiled(torch.zeros(_NUM_ENVS, dtype=torch.int32))
    env.close()
    assert len(result) == 8


def test_pending_action_space():
    """allow_pending_action grows each env's action space by one.

    The extra last id is the pending action; consumers derive it as
    nvec[i] - 1. With the flag off the space is unchanged.
    """
    env_off = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_on = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, allow_pending_action=True)
    n = int(env_off.action_space.nvec[0])
    n_on = int(env_on.action_space.nvec[0])
    env_off.close()
    env_on.close()
    assert n_on == n + 1


@cpp_required
def test_cpp_pending_action_flag_no_side_effect():
    """The flag alone does not perturb dynamics.

    A flag-on env stepping only real actions is byte-identical to a flag-off
    env (the plain two-argument step) with the same seed and actions.
    """
    env_off = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS)
    env_on = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, allow_pending_action=True)
    env_off.reset(seed=0)
    env_on.reset(seed=0)
    env_off.torch(cpp=True)
    env_on.torch(cpp=True)

    rng = np.random.default_rng(0)
    for _ in range(10):
        acts = torch.from_numpy(rng.integers(0, 6, size=_NUM_ENVS, dtype=np.int32))
        for a, b in zip(env_off.step(acts), env_on.step(acts)):
            assert torch.equal(a, b)

    env_off.close()
    env_on.close()


@cpp_required
def test_cpp_pending_action_holds_env():
    """The pending action freezes its env while others advance.

    Both envs settle past the reset noops with real actions. Sending the
    pending action to env 0 then holds it: byte-identical observation and
    episode_frame_number, reward -inf (the sentinel for unobserved, written in
    place of the clip), terminated/truncated False. Env 1 keeps acting and its
    reward stays clipped to [-1, 1]. Ids past the pending action are invalid,
    and a real action advances the held env again.
    """
    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, allow_pending_action=True)
    env.reset(seed=0)
    env.torch(cpp=True)
    pending_id = int(env.action_space.nvec[0]) - 1
    actions = torch.full((_NUM_ENVS,), 2, dtype=torch.int32)
    for _ in range(20):
        obs, *_ = env.step(actions)

    hold_first = torch.tensor([pending_id, 2], dtype=torch.int32)
    frozen = obs[0]
    ep_frame_frozen = None
    for _ in range(3):
        obs, reward, term, trunc, _, _, _, ep_frame, *_ = env.step(hold_first)
        assert torch.equal(obs[0], frozen)
        assert torch.isneginf(reward[0])
        assert not term[0] and not trunc[0]
        if ep_frame_frozen is None:
            ep_frame_frozen = ep_frame[0]
        assert ep_frame[0] == ep_frame_frozen
        assert torch.isfinite(reward[1]) and reward[1].abs() <= 1
    assert not torch.equal(obs[1], frozen)

    with pytest.raises(Exception):
        env.step(torch.tensor([pending_id + 1, 2], dtype=torch.int32))

    obs, *_ = env.step(actions)
    changed = not torch.equal(obs[0], frozen)
    env.close()
    assert changed


@cpp_required
def test_cpp_pending_action_compile():
    """The pending action flows through torch.compile and scan.

    A fullgraph-compiled step with the pending action returns the -inf
    sentinel for the held env, and a scan rollout holding env 0 throughout
    traces without graph breaks and returns -inf at every held step.
    """
    from torch._higher_order_ops import scan

    env = AtariVectorEnv(_GAME, num_envs=_NUM_ENVS, allow_pending_action=True)
    env.reset(seed=0)
    env.torch(cpp=True)
    pending_id = int(env.action_space.nvec[0]) - 1
    hold_first = torch.tensor([pending_id, 2], dtype=torch.int32)

    compiled = torch.compile(env.step, fullgraph=True)
    out = compiled(hold_first)
    obs, reward = out[0], out[1]
    assert len(out) == 9 and torch.isneginf(reward[0])

    def step_fn(carry, actions):
        obs, reward, *_ = env.step(actions)
        return obs, reward

    xs = hold_first.expand(4, _NUM_ENVS).contiguous()
    _, rewards = scan(step_fn, obs, xs)
    env.close()
    assert rewards.shape == (4, _NUM_ENVS)
    assert torch.isneginf(rewards[:, 0]).all()
