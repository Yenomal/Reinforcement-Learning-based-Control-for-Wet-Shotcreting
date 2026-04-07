#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal PPO implementation for the mathematical planning environment."""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn
from torch.distributions import Normal

from ..component.buffer import OnPolicyBatch
from ..model.mlp import MLP


class PPOAgent:
    """PPO agent with separate actor and critic MLPs."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        model_cfg: Dict[str, Any],
        algorithm_cfg: Dict[str, Any],
        device: torch.device,
    ) -> None:
        self.device = device

        hidden_sizes = model_cfg.get("hidden_sizes", [256, 256])
        activation = str(model_cfg.get("activation", "tanh"))
        init_log_std = float(model_cfg.get("init_log_std", -0.5))

        self.actor = MLP(
            input_dim=observation_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.critic = MLP(
            input_dim=observation_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, device=self.device)
        )

        parameters = list(self.actor.parameters()) + list(self.critic.parameters()) + [
            self.log_std
        ]
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=float(algorithm_cfg.get("lr", 3.0e-4)),
        )

        self.gamma = float(algorithm_cfg.get("gamma", 0.99))
        self.gae_lambda = float(algorithm_cfg.get("gae_lambda", 0.95))
        self.clip_ratio = float(algorithm_cfg.get("clip_ratio", 0.2))
        self.value_coef = float(algorithm_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(algorithm_cfg.get("entropy_coef", 0.0))
        self.max_grad_norm = float(algorithm_cfg.get("max_grad_norm", 0.5))
        self.update_epochs = int(algorithm_cfg.get("update_epochs", 10))
        self.minibatch_size = int(algorithm_cfg.get("minibatch_size", 256))
        self.normalize_advantages = bool(
            algorithm_cfg.get("normalize_advantages", True)
        )

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distribution = self._distribution(observation)
            action = distribution.sample()
            log_prob = distribution.log_prob(action).sum(dim=-1)
            value = self.critic(observation).squeeze(-1)
        return action, log_prob, value

    def act_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action mean for evaluation."""
        with torch.no_grad():
            return self.actor(observation)

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.critic(observation).squeeze(-1)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value = self.critic(observations).squeeze(-1)
        return log_prob, entropy, value

    def compute_returns_and_advantages(
        self, batch: OnPolicyBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = batch.rewards
        dones = batch.dones
        values = batch.values

        advantages = torch.zeros_like(rewards, device=self.device)
        last_gae = torch.zeros(1, device=self.device)

        next_value = self.value(batch.next_observation.unsqueeze(0)).squeeze(0)
        next_done = batch.next_done.squeeze(0)

        for step in reversed(range(rewards.shape[0])):
            if step == rewards.shape[0] - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step]
                next_values = values[step + 1]

            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[step] = last_gae

        returns = advantages + values
        return returns, advantages

    def update(self, batch: OnPolicyBatch) -> Dict[str, float]:
        returns, advantages = self.compute_returns_and_advantages(batch)

        observations = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = observations.shape[0]
        indices = torch.arange(batch_size, device=self.device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0

        for _ in range(self.update_epochs):
            permutation = indices[torch.randperm(batch_size, device=self.device)]

            for start in range(0, batch_size, self.minibatch_size):
                minibatch_indices = permutation[start : start + self.minibatch_size]

                mb_obs = observations[minibatch_indices]
                mb_actions = actions[minibatch_indices]
                mb_old_log_probs = old_log_probs[minibatch_indices]
                mb_advantages = advantages[minibatch_indices]
                mb_returns = returns[minibatch_indices]

                new_log_probs, entropy, values = self.evaluate_actions(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * torch.mean((values - mb_returns) ** 2)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters())
                    + [self.log_std],
                    self.max_grad_norm,
                )
                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_bonus.item())
                last_kl = float((mb_old_log_probs - new_log_probs).mean().item())

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
            "returns_mean": float(returns.mean().item()),
            "advantages_mean": float(advantages.mean().item()),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_policy_state(self, state_dict: Dict[str, Any]) -> None:
        """Load actor, critic, and log_std from a checkpoint payload."""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        with torch.no_grad():
            self.log_std.copy_(state_dict["log_std"].to(self.device))

        self.actor.eval()
        self.critic.eval()
