#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Soft Actor-Critic implementation for the mathematical planning environment."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ..component.buffer import ReplayBatch
from ..model.mlp import MLP


LOG_STD_EPS = 1e-6

DEFAULT_SAC_CONFIG: Dict[str, Any] = {
    "actor_lr": 3.0e-4,
    "critic_lr": 3.0e-4,
    "alpha_lr": 3.0e-4,
    "tau": 0.005,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "learning_starts": 100,
    "updates_per_step": 1,
    "log_interval_steps": 2000,
    "alpha_init": 1.0,
    "target_entropy": "auto",
    "log_std_min": -5.0,
    "log_std_max": 2.0,
}


def build_sac_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build SAC hyperparameters from the project defaults."""
    resolved = dict(DEFAULT_SAC_CONFIG)
    if overrides is None:
        return resolved

    for key in DEFAULT_SAC_CONFIG:
        if key in overrides:
            resolved[key] = overrides[key]
    return resolved


class SACAgent:
    """Continuous-control SAC agent with twin critics."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        model_cfg: Dict[str, Any],
        algorithm_cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.device = device
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        resolved_cfg = build_sac_config(algorithm_cfg)

        hidden_sizes = model_cfg.get("hidden_sizes", [256, 256])
        activation = str(model_cfg.get("activation", "tanh"))
        self.log_std_min = float(resolved_cfg["log_std_min"])
        self.log_std_max = float(resolved_cfg["log_std_max"])

        self.actor = MLP(
            input_dim=self.observation_dim,
            output_dim=self.action_dim * 2,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        critic_input_dim = self.observation_dim + self.action_dim
        self.critic_1 = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.critic_2 = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.target_critic_1 = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.target_critic_2 = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.eval()
        self.target_critic_2.eval()

        self.gamma = float(algorithm_cfg.get("gamma", 0.99))
        self.tau = float(resolved_cfg["tau"])

        actor_lr = float(resolved_cfg["actor_lr"])
        critic_lr = float(resolved_cfg["critic_lr"])
        alpha_lr = float(resolved_cfg["alpha_lr"])

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        alpha_init = float(resolved_cfg["alpha_init"])
        self.log_alpha = torch.tensor(
            np.log(alpha_init),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        target_entropy = resolved_cfg["target_entropy"]
        if isinstance(target_entropy, str) and target_entropy.lower() == "auto":
            self.target_entropy = -float(self.action_dim)
        else:
            self.target_entropy = float(target_entropy)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _actor_stats(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.actor(observations)
        mean, log_std = torch.chunk(stats, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def _sample_action(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._actor_stats(observations)
        std = log_std.exp()
        distribution = Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = distribution.log_prob(pre_tanh).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + LOG_STD_EPS).sum(dim=-1)
        log_prob = log_prob - correction
        return action, log_prob

    def _critic_forward(
        self, critic: nn.Module, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        critic_input = torch.cat([observations, actions], dim=-1)
        return critic(critic_input).squeeze(-1)

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self._sample_action(observation)
        return action

    def act_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mean, _ = self._actor_stats(observation)
            return torch.tanh(mean)

    def update(self, batch: ReplayBatch) -> Dict[str, float]:
        observations = batch.observations
        actions = batch.actions
        rewards = batch.rewards
        next_observations = batch.next_observations
        dones = batch.dones

        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_observations)
            target_q1 = self._critic_forward(
                self.target_critic_1, next_observations, next_actions
            )
            target_q2 = self._critic_forward(
                self.target_critic_2, next_observations, next_actions
            )
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            target_values = rewards + self.gamma * (1.0 - dones) * target_q

        current_q1 = self._critic_forward(self.critic_1, observations, actions)
        current_q2 = self._critic_forward(self.critic_2, observations, actions)

        critic_1_loss = torch.mean((current_q1 - target_values) ** 2)
        critic_2_loss = torch.mean((current_q2 - target_values) ** 2)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        sampled_actions, sampled_log_probs = self._sample_action(observations)
        q1_pi = self._critic_forward(self.critic_1, observations, sampled_actions)
        q2_pi = self._critic_forward(self.critic_2, observations, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = torch.mean(self.alpha.detach() * sampled_log_probs - q_pi)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -torch.mean(
            self.log_alpha * (sampled_log_probs.detach() + self.target_entropy)
        )
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic_1, self.target_critic_1)
        self._soft_update(self.critic_2, self.target_critic_2)

        return {
            "policy_loss": float(actor_loss.item()),
            "value_loss": float(critic_loss.item()),
            "critic_1_loss": float(critic_1_loss.item()),
            "critic_2_loss": float(critic_2_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "q_mean": float(q_pi.mean().detach().cpu().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_policy_state(self, state_dict: Dict[str, Any]) -> None:
        self.actor.load_state_dict(state_dict["actor"])

        if "critic_1" in state_dict:
            self.critic_1.load_state_dict(state_dict["critic_1"])
        if "critic_2" in state_dict:
            self.critic_2.load_state_dict(state_dict["critic_2"])
        if "target_critic_1" in state_dict:
            self.target_critic_1.load_state_dict(state_dict["target_critic_1"])
        if "target_critic_2" in state_dict:
            self.target_critic_2.load_state_dict(state_dict["target_critic_2"])
        if "log_alpha" in state_dict:
            with torch.no_grad():
                self.log_alpha.copy_(state_dict["log_alpha"].to(self.device))

        self.actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()
        self.target_critic_1.eval()
        self.target_critic_2.eval()

    def load_training_state(self, state_dict: Dict[str, Any]) -> None:
        """Load full SAC training state, including optimizers."""
        self.load_policy_state(state_dict)
        if "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if "critic_1_optimizer" in state_dict:
            self.critic_1_optimizer.load_state_dict(state_dict["critic_1_optimizer"])
        if "critic_2_optimizer" in state_dict:
            self.critic_2_optimizer.load_state_dict(state_dict["critic_2_optimizer"])
        if "alpha_optimizer" in state_dict:
            self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        self.actor.train()
        self.critic_1.train()
        self.critic_2.train()
