"""TorchRL wrapper for AtariVectorEnv - constructed via AtariVectorEnv.torchrl().

Wraps the vectorized ALE environment as a ``torchrl.envs.EnvBase`` so it plugs
straight into TorchRL collectors, replay buffers and trainers. The heavy lifting
(zero-copy tensors, pinned-memory transfers, optional CUDA placement) is done by
the existing ``env.torch(tensordict=True)`` path; this module only adds the
TorchRL spec/contract layer on top.

Do not import this module at the top level; it is lazy-loaded by torchrl().
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Bounded,
    BoundedContinuous,
    Categorical,
    Composite,
    MultiCategorical,
    Unbounded,
)
from torchrl.envs import EnvBase

__all__ = ["AtariTorchRLEnv"]

_INFO_KEYS = ("env_id", "lives", "frame_number", "episode_frame_number")


class AtariTorchRLEnv(EnvBase):
    """A ``torchrl.envs.EnvBase`` view over :class:`AtariVectorEnv`.

    The batch dimension is the number of sub-environments, so the env behaves
    like a TorchRL batched env with ``batch_size = [num_envs]``. Observations
    are uint8 frame stacks; the ALE info fields (``env_id``, ``lives``,
    ``frame_number``, ``episode_frame_number``) are exposed as extra entries of
    the observation spec.
    """

    # We manage batching ourselves (one ALE vectorizer covers the whole batch),
    # so TorchRL must not try to add its own batch dimension.
    batch_locked = True

    def __init__(self, env, device=None):
        """Wrap an ``AtariVectorEnv`` as a TorchRL environment.

        Args:
            env: The ``AtariVectorEnv`` to wrap. Its step/send/recv are patched
                to exchange tensors on ``device``.
            device: Target device for observations, actions, and rewards.
        """
        num_envs = env.num_envs
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self._env = env
        # Patch step/send/recv to speak tensors + TensorDicts on the target device.
        env.torch(device=device, tensordict=True)
        self._setup_specs()

    # ------------------------------------------------------------------ specs
    def _setup_specs(self) -> None:
        env = self._env
        num_envs = env.num_envs
        dev = self.device

        single_obs_shape = env.single_observation_space.shape
        assert single_obs_shape is not None
        obs_shape = (num_envs, *single_obs_shape)

        self.observation_spec = Composite(
            {
                "obs": Bounded(
                    low=0, high=255, shape=obs_shape, dtype=torch.uint8, device=dev
                ),
                **{
                    k: Unbounded(shape=(num_envs,), dtype=torch.int32, device=dev)
                    for k in _INFO_KEYS
                },
            },
            shape=(num_envs,),
            device=dev,
        )

        if env.continuous:
            single = env.single_action_space
            low = torch.as_tensor(np.asarray(single.low), dtype=torch.float32)
            high = torch.as_tensor(np.asarray(single.high), dtype=torch.float32)
            self.action_spec = BoundedContinuous(
                low=low.expand(num_envs, 3).clone(),
                high=high.expand(num_envs, 3).clone(),
                shape=(num_envs, 3),
                dtype=torch.float32,
                device=dev,
            )
        else:
            nvec = np.asarray(env.action_space.nvec)
            if bool(np.all(nvec == nvec[0])):
                self.action_spec = Categorical(
                    n=int(nvec[0]), shape=(num_envs,), dtype=torch.int64, device=dev
                )
            else:
                # Heterogeneous games: each env keeps its own action-set size.
                self.action_spec = MultiCategorical(
                    nvec=nvec.tolist(), dtype=torch.int64, device=dev
                )

        self.reward_spec = Unbounded(
            shape=(num_envs, 1), dtype=torch.float32, device=dev
        )

        self.full_done_spec = Composite(
            {
                k: Categorical(n=2, shape=(num_envs, 1), dtype=torch.bool, device=dev)
                for k in ("done", "terminated", "truncated")
            },
            shape=(num_envs,),
            device=dev,
        )

    # --------------------------------------------------------------- contract
    def _to_next_td(self, result: TensorDictBase) -> TensorDict:
        """Convert an ALE step TensorDict into a TorchRL ``next`` TensorDict."""
        reward = result["reward"].to(torch.float32).reshape(*self.batch_size, 1)
        terminated = result["term"].reshape(*self.batch_size, 1)
        truncated = result["trunc"].reshape(*self.batch_size, 1)
        done = terminated | truncated
        return TensorDict(
            {
                "obs": result["obs"],
                **{k: result[k] for k in _INFO_KEYS},
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "done": done,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        result = self._env.step(tensordict["action"])
        return self._to_next_td(result)

    def _reset(self, tensordict: TensorDictBase | None = None, **kwargs) -> TensorDict:
        seed = getattr(self, "_next_seed", None)
        self._next_seed = None
        # Use the unpatched reset: env.reset is patched to return tensors after
        # .torch(), but here we want the raw numpy arrays to build the spec td.
        obs, info = type(self._env).reset(self._env, seed=seed)
        return TensorDict(
            {
                "obs": torch.from_numpy(obs),
                **{k: torch.from_numpy(info[k]) for k in _INFO_KEYS},
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed: int | None) -> None:
        # Consumed by the next _reset(); ALE seeds per-environment on reset.
        self._next_seed = seed

    # ------------------------------------------------ async step (send / recv)
    # Mirrors torchrl's async env interface: split a step into a non-blocking
    # send followed by a later recv, so the policy can run while ALE steps.
    def async_step_send(self, tensordict: TensorDictBase) -> None:
        """Dispatch the batch of actions without waiting for results."""
        self._env.send(tensordict["action"])

    def async_step_recv(self) -> TensorDict:
        """Collect the results of the previously sent step as a ``next`` td."""
        return self._to_next_td(self._env.recv())

    def close(self, *args, **kwargs) -> None:
        """Close the underlying ALE vector env (unregistering its torch ops)."""
        self._env.close()
        super().close(*args, **kwargs)
