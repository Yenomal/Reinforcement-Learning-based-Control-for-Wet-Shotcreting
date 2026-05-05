#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal PPO implementation for the mathematical planning environment."""

from __future__ import annotations

import copy
from typing import Any, Dict

import torch
from torch import nn
from torch.distributions import Normal

from ..component.buffer import OnPolicyBatch
from ..model.mlp import build_state_network


SQUASH_EPS = 1.0e-6

DEFAULT_PPO_CONFIG: Dict[str, Any] = {
    "lr": 3.0e-4,  # 优化器学习率
    "init_log_std": -2.0,  # 旧版全局 std 初值
    "gae_lambda": 0.95,  # GAE 衰减系数
    "clip_ratio": 0.2,  # PPO clip 范围
    "value_coef": 0.1,  # value loss 权重
    "normalize_value_targets": False,  # 是否归一化 value target
    "entropy_coef": 0.0,  # 熵奖励权重
    "max_grad_norm": 0.5,  # 梯度裁剪上限
    "rollout_steps": 65536,  # 每轮采样总步数
    "update_epochs": 10,  # 每轮 PPO 更新次数
    "minibatch_size": 512,  # PPO 小批大小
    "normalize_advantages": True,  # 是否标准化优势
    "std": {
        "mode": "cosine_schedule",  # std 模式
        "global_log_std": -2.0,  # 全局初值或固定值
        "min_log_std": -6.0,  # 允许下降到的最小 log_std
        "switch_update": 300,  # 组合模式切换轮次
        "schedule": {
            "schedule": "cosine",  # std 调度曲线
            "start_log_std": -1.0,  # 调度起点
            "end_log_std": -6.0,  # 调度终点
        },
        "success_trigger": {
            "success_threshold": 0.90,  # EMA 成功率阈值
            "log_std_step": -0.5,  # 每次下降的 log_std 步长
            "ema_alpha": 0.1,  # 成功率 EMA 平滑系数
            "patience_updates": 10,  # 连续满足多少轮才下降
        },
    },
}


def build_ppo_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build PPO hyperparameters from the project defaults."""
    def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(base.get(key), dict) and isinstance(value, dict):
                _deep_update(base[key], value)
            else:
                base[key] = value

    resolved = copy.deepcopy(DEFAULT_PPO_CONFIG)
    if overrides is None:
        return resolved

    for key in DEFAULT_PPO_CONFIG:
        if key in overrides:
            if (
                isinstance(DEFAULT_PPO_CONFIG[key], dict)
                and isinstance(overrides[key], dict)
            ):
                _deep_update(resolved[key], overrides[key])
            else:
                resolved[key] = overrides[key]
    return resolved


def resolve_ppo_std_config(algorithm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve PPO std behavior from the current config and legacy fields."""
    resolved_cfg = build_ppo_config(algorithm_cfg)
    std_cfg = copy.deepcopy(resolved_cfg["std"])
    # 兼容旧版 exploration_schedule 配置。
    legacy_schedule_cfg = dict(algorithm_cfg.get("exploration_schedule", {}))
    raw_std_cfg = algorithm_cfg.get("std")

    if raw_std_cfg is None and "exploration_schedule" in algorithm_cfg:
        legacy_schedule_enabled = bool(legacy_schedule_cfg.get("enable", False))
        if legacy_schedule_enabled:
            std_cfg["mode"] = "cosine_schedule"
            std_cfg["schedule"].update(
                {
                    "schedule": str(
                        legacy_schedule_cfg.get(
                            "schedule",
                            std_cfg["schedule"].get("schedule", "cosine"),
                        )
                    ),
                    "start_log_std": float(
                        legacy_schedule_cfg.get(
                            "start_log_std",
                            std_cfg["schedule"].get(
                                "start_log_std",
                                std_cfg.get("global_log_std", resolved_cfg["init_log_std"]),
                            ),
                        )
                    ),
                    "end_log_std": float(
                        legacy_schedule_cfg.get(
                            "end_log_std",
                            std_cfg["schedule"].get(
                                "end_log_std",
                                std_cfg.get("global_log_std", resolved_cfg["init_log_std"]),
                            ),
                        )
                    ),
                }
            )
        else:
            std_cfg["mode"] = "global_learned"

    std_cfg["mode"] = str(std_cfg.get("mode", "cosine_schedule")).lower()
    if "global_log_std" not in std_cfg:
        std_cfg["global_log_std"] = float(resolved_cfg["init_log_std"])
    std_cfg["global_log_std"] = float(std_cfg["global_log_std"])
    std_cfg["min_log_std"] = float(std_cfg.get("min_log_std", -6.0))
    if std_cfg["min_log_std"] > std_cfg["global_log_std"]:
        raise ValueError("ppo.std.min_log_std must be <= ppo.std.global_log_std.")
    std_cfg["switch_update"] = max(int(std_cfg.get("switch_update", 0)), 0)
    std_cfg["schedule"] = dict(std_cfg.get("schedule", {}))
    std_cfg["success_trigger"] = dict(std_cfg.get("success_trigger", {}))
    return std_cfg


class RunningValueNormalizer:
    """Running mean/std normalizer for value targets."""

    def __init__(self, eps: float = 1.0e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = eps
        self.eps = eps

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().flatten()
        if values.numel() == 0:
            return

        batch_mean = float(values.mean().cpu().item())
        batch_var = float(values.var(unbiased=False).cpu().item())
        batch_count = float(values.numel())
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: float, batch_var: float, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = max(new_var, self.eps)
        self.count = total_count

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(
            torch.tensor(self.var + self.eps, dtype=values.dtype, device=values.device)
        )
        mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device)
        return (values - mean) / scale

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        scale = torch.sqrt(
            torch.tensor(self.var + self.eps, dtype=values.dtype, device=values.device)
        )
        mean = torch.tensor(self.mean, dtype=values.dtype, device=values.device)
        return values * scale + mean

    def state_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.mean = float(state_dict.get("mean", 0.0))
        self.var = float(state_dict.get("var", 1.0))
        self.count = float(state_dict.get("count", self.eps))
        self.eps = float(state_dict.get("eps", self.eps))


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
        resolved_cfg = build_ppo_config(algorithm_cfg)
        std_cfg = resolve_ppo_std_config(algorithm_cfg)
        self.std_mode = str(std_cfg["mode"]).lower()
        valid_std_modes = {
            "cosine_schedule",
            "cosine_then_success_rate_triggered",
            "success_rate_triggered",
            "global_fixed",
            "global_learned",
        }
        if self.std_mode not in valid_std_modes:
            raise ValueError(
                "Unsupported PPO std mode: "
                f"{self.std_mode}. Expected one of {sorted(valid_std_modes)}."
            )
        self.global_log_std = float(std_cfg["global_log_std"])
        self.log_std_min = float(std_cfg["min_log_std"])

        self.actor = build_state_network(
            input_dim=observation_dim,
            output_dim=action_dim,
            model_cfg=model_cfg,
        ).to(self.device)
        self.critic = build_state_network(
            input_dim=observation_dim,
            output_dim=1,
            model_cfg=model_cfg,
        ).to(self.device)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), self.global_log_std, device=self.device)
        )
        self.log_std.requires_grad_(self.std_mode == "global_learned")

        parameters = list(self.actor.parameters())
        if self.log_std is not None and self.log_std.requires_grad:
            parameters.append(self.log_std)
        parameters.extend(self.critic.parameters())
        self.optimizer = torch.optim.Adam(
            parameters,
            lr=float(resolved_cfg["lr"]),
        )

        self.gamma = float(algorithm_cfg.get("gamma", 0.99))
        self.gae_lambda = float(resolved_cfg["gae_lambda"])
        self.clip_ratio = float(resolved_cfg["clip_ratio"])
        self.value_coef = float(resolved_cfg["value_coef"])
        self.entropy_coef = float(resolved_cfg["entropy_coef"])
        self.max_grad_norm = float(resolved_cfg["max_grad_norm"])
        self.update_epochs = int(resolved_cfg["update_epochs"])
        self.minibatch_size = int(resolved_cfg["minibatch_size"])
        self.normalize_advantages = bool(resolved_cfg["normalize_advantages"])
        self.normalize_value_targets = bool(resolved_cfg["normalize_value_targets"])
        self.value_normalizer = RunningValueNormalizer()

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        log_std = self.log_std.view(1, -1).expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def set_log_std_value(self, value: float) -> None:
        with torch.no_grad():
            clamped_value = max(float(value), self.log_std_min)
            self.log_std.fill_(clamped_value)

    def get_log_std_value(self) -> float:
        return float(self.log_std.detach().mean().cpu().item())

    def get_std_value(self) -> float:
        return float(torch.exp(self.log_std.detach()).mean().cpu().item())

    @staticmethod
    def _squash_action(pre_tanh_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh_action)

    @staticmethod
    def _unsquash_action(action: torch.Tensor) -> torch.Tensor:
        clipped_action = torch.clamp(action, -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
        return 0.5 * (
            torch.log1p(clipped_action) - torch.log1p(-clipped_action)
        )

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal,
        pre_tanh_action: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = distribution.log_prob(pre_tanh_action).sum(dim=-1)
        correction = torch.log(1.0 - action.pow(2) + SQUASH_EPS).sum(dim=-1)
        return log_prob - correction

    def _critic_normalized(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation).squeeze(-1)

    def act(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            distribution = self._distribution(observation)
            pre_tanh_action = distribution.sample()
            action = self._squash_action(pre_tanh_action)
            log_prob = self._squashed_log_prob(distribution, pre_tanh_action, action)
            value = self._critic_normalized(observation)
            if self.normalize_value_targets:
                value = self.value_normalizer.denormalize(value)
        return action, log_prob, value

    def act_deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the deterministic bounded action mean for evaluation."""
        with torch.no_grad():
            return self._squash_action(self.actor(observation))

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            value = self._critic_normalized(observation)
            if self.normalize_value_targets:
                value = self.value_normalizer.denormalize(value)
            return value

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self._distribution(observations)
        clipped_actions = torch.clamp(actions, -1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
        pre_tanh_action = self._unsquash_action(clipped_actions)
        log_prob = self._squashed_log_prob(
            distribution,
            pre_tanh_action,
            clipped_actions,
        )
        # The exact tanh-squashed entropy has no simple closed form here;
        # we keep the base Normal entropy as a stable proxy.
        entropy = distribution.entropy().sum(dim=-1)
        normalized_value = self._critic_normalized(observations)
        value = normalized_value
        if self.normalize_value_targets:
            value = self.value_normalizer.denormalize(normalized_value)
        return log_prob, entropy, value, normalized_value

    def compute_returns_and_advantages(
        self, batch: OnPolicyBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = batch.rewards
        dones = batch.dones
        values = batch.values

        advantages = torch.zeros_like(rewards, device=self.device)
        if rewards.ndim == 1:
            last_gae = torch.zeros(1, device=self.device)
            next_value = self.value(batch.next_observation.unsqueeze(0)).squeeze(0)
            next_done = batch.next_done.reshape(1).squeeze(0)
        else:
            last_gae = torch.zeros(rewards.shape[1], device=self.device)
            next_value = self.value(batch.next_observation)
            next_done = batch.next_done

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
        value_predictions = batch.values

        observations = batch.observations
        actions = batch.actions
        old_log_probs = batch.log_probs

        flat_returns = returns.reshape(-1)
        flat_value_predictions = value_predictions.reshape(-1)
        returns_var = torch.var(flat_returns, unbiased=False)
        if float(returns_var.detach().cpu().item()) <= 1.0e-12:
            explained_variance = float("nan")
        else:
            residual_var = torch.var(
                flat_returns - flat_value_predictions,
                unbiased=False,
            )
            explained_variance = float(
                (1.0 - residual_var / returns_var).detach().cpu().item()
            )

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        normalized_returns = returns
        if self.normalize_value_targets:
            self.value_normalizer.update(returns)
            normalized_returns = self.value_normalizer.normalize(returns)

        if observations.ndim > 2:
            observation_dim = observations.shape[-1]
            action_dim = actions.shape[-1]
            observations = observations.reshape(-1, observation_dim)
            actions = actions.reshape(-1, action_dim)
            old_log_probs = old_log_probs.reshape(-1)
            returns = returns.reshape(-1)
            advantages = advantages.reshape(-1)
            normalized_returns = normalized_returns.reshape(-1)

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
                mb_normalized_returns = normalized_returns[minibatch_indices]

                new_log_probs, entropy, values, normalized_values = self.evaluate_actions(
                    mb_obs, mb_actions
                )
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                if self.normalize_value_targets:
                    value_loss = 0.5 * torch.mean(
                        (normalized_values - mb_normalized_returns) ** 2
                    )
                else:
                    value_loss = 0.5 * torch.mean((values - mb_returns) ** 2)
                entropy_bonus = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
                if self.log_std is not None and self.log_std.requires_grad:
                    grad_parameters.append(self.log_std)
                nn.utils.clip_grad_norm_(grad_parameters, self.max_grad_norm)
                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_bonus.item())
                last_kl = float((mb_old_log_probs - new_log_probs).mean().item())

        log_std_mean = self.get_log_std_value()
        std_mean = self.get_std_value()

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
            "explained_variance": explained_variance,
            "returns_mean": float(returns.mean().item()),
            "normalized_returns_mean": float(normalized_returns.mean().item()),
            "advantages_mean": float(advantages.mean().item()),
            "ppo_log_std_mean": log_std_mean,
            "ppo_std_mean": std_mean,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "std_mode": self.std_mode,
            "log_std": self.log_std.detach().cpu(),
        }

    def load_policy_state(self, state_dict: Dict[str, Any]) -> None:
        """Load actor, critic, and log_std from a checkpoint payload."""
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        if "log_std" not in state_dict:
            raise KeyError("Checkpoint does not contain global PPO log_std weights.")
        with torch.no_grad():
            self.log_std.copy_(state_dict["log_std"].to(self.device))
        if "value_normalizer" in state_dict:
            self.value_normalizer.load_state_dict(state_dict["value_normalizer"])

        self.actor.eval()
        self.critic.eval()

    def load_training_state(self, state_dict: Dict[str, Any]) -> None:
        """Load full PPO training state, including optimizer."""
        self.load_policy_state(state_dict)
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        self.actor.train()
        self.critic.train()
