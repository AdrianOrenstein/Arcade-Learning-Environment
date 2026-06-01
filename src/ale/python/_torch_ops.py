"""PyTorch custom ops for ALE - registered via AtariVectorEnv.torch().

No 'from __future__ import annotations' here so torch.library.custom_op can
infer op schemas from type annotations without PEP 563 string-ification.

Do not import this module at the top level; it is lazy-loaded by torch().
"""

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch._library.effects import EffectType

if TYPE_CHECKING:
    from ale_py.vector_env import AtariVectorEnv

__all__ = ["register_pytorch_ops"]

_torch_registered: bool = False
_torch_buffers: dict[int, dict] = {}


def register_pytorch_ops(env: "AtariVectorEnv"):
    """Allocate pinned buffers and register ale::* torch custom ops once.

    Returns:
        (handle_id, ale_send, ale_step, ale_recv, unregister)
    """
    global _torch_registered

    handle_id = id(env)
    num_envs = env.num_envs
    assert env.single_observation_space.shape is not None
    obs_shape = (num_envs,) + env.single_observation_space.shape

    _torch_buffers[handle_id] = {
        "vector_interface": env.ale,
        "actions": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "obs": torch.empty(obs_shape, dtype=torch.uint8, pin_memory=True),
        "reward": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
        "term": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "trunc": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "paddle_strength": np.ones(num_envs, dtype=np.float32),
        "actions_d2h_event": torch.cuda.Event() if torch.cuda.is_available() else None,
    }

    if not _torch_registered:
        _torch_registered = True

        @torch.library.custom_op("ale::send", mutates_args=())
        @torch.no_grad()
        def ale_send(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            buf["actions"].copy_(actions, non_blocking=True)
            if actions.is_cuda:
                buf["actions_d2h_event"].record()
                buf["actions_d2h_event"].synchronize()
            buf["vector_interface"].send(buf["actions"].numpy(), buf["paddle_strength"])
            return actions.new_empty(())

        ale_send.register_effect(EffectType.ORDERED)

        @ale_send.register_fake
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty(())

        @torch.library.custom_op("ale::recv", mutates_args=())
        def ale_recv(
            handle_id: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            buf = _torch_buffers[handle_id]
            obs, reward, term, trunc, info = buf["vector_interface"].recv()
            buf["obs"].copy_(torch.from_numpy(obs))
            buf["reward"].copy_(torch.from_numpy(reward))
            buf["term"].copy_(torch.from_numpy(term))
            buf["trunc"].copy_(torch.from_numpy(trunc))
            buf["last_info"] = info
            return (buf["obs"], buf["reward"], buf["term"], buf["trunc"])

        ale_recv.register_effect(EffectType.ORDERED)

        @ale_recv.register_fake
        def _(
            handle_id: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            buf = _torch_buffers[handle_id]
            return (
                torch.empty(buf["obs"].shape, dtype=buf["obs"].dtype),
                torch.empty(buf["reward"].shape, dtype=buf["reward"].dtype),
                torch.empty(buf["term"].shape, dtype=buf["term"].dtype),
                torch.empty(buf["trunc"].shape, dtype=buf["trunc"].dtype),
            )

    def unregister() -> None:
        _torch_buffers.pop(handle_id, None)

    def ale_step(
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        torch.ops.ale.send(handle_id, actions)
        obs, reward, term, trunc = torch.ops.ale.recv(handle_id)
        info = _torch_buffers[handle_id].get("last_info", {})
        return obs, reward, term, trunc, info

    return (
        handle_id,
        torch.ops.ale.send,
        ale_step,
        torch.ops.ale.recv,
        unregister,
    )
