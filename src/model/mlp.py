#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared MLP building blocks."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    """Create an activation module from a config string."""
    normalized = name.lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """Simple multi-layer perceptron."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(build_activation(activation))
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _normalize_hidden_sizes(hidden_sizes: Iterable[int] | None) -> list[int]:
    if hidden_sizes is None:
        return []
    return [int(hidden_size) for hidden_size in hidden_sizes]


def _build_branch(
    input_dim: int,
    hidden_sizes: Iterable[int] | None,
    activation: str,
) -> tuple[nn.Module, int]:
    resolved_hidden_sizes = _normalize_hidden_sizes(hidden_sizes)
    if not resolved_hidden_sizes:
        return nn.Identity(), int(input_dim)
    return (
        MLP(
            input_dim=input_dim,
            output_dim=resolved_hidden_sizes[-1],
            hidden_sizes=resolved_hidden_sizes[:-1],
            activation=activation,
        ),
        resolved_hidden_sizes[-1],
    )


class StructuredObservationEncoder(nn.Module):
    """Encode grouped observation fields with separate MLP branches."""

    JOINT_DIM = 4
    GEOMETRY_DIM = 9
    PREV_ACTION_DIM = 4
    TIME_DIM = 1
    OBSERVATION_DIM = JOINT_DIM + GEOMETRY_DIM + PREV_ACTION_DIM + TIME_DIM

    def __init__(self, input_dim: int, model_cfg: dict[str, Any]) -> None:
        super().__init__()
        if int(input_dim) != self.OBSERVATION_DIM:
            raise ValueError(
                "StructuredObservationEncoder expects observation_dim=18, "
                f"got {input_dim}."
            )

        activation = str(model_cfg.get("activation", "tanh"))
        self.joint_encoder, joint_dim = _build_branch(
            self.JOINT_DIM,
            model_cfg.get("joint_hidden_sizes", [128, 128]),
            activation,
        )
        self.geometry_encoder, geometry_dim = _build_branch(
            self.GEOMETRY_DIM,
            model_cfg.get("geometry_hidden_sizes", [128, 128]),
            activation,
        )
        self.prev_action_encoder, prev_action_dim = _build_branch(
            self.PREV_ACTION_DIM,
            model_cfg.get("prev_action_hidden_sizes", [64, 64]),
            activation,
        )
        self.time_encoder, time_dim = _build_branch(
            self.TIME_DIM,
            model_cfg.get("time_hidden_sizes", [32, 32]),
            activation,
        )
        self.output_dim = int(joint_dim + geometry_dim + prev_action_dim + time_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        joints = inputs[..., : self.JOINT_DIM]
        geometry = inputs[
            ...,
            self.JOINT_DIM : self.JOINT_DIM + self.GEOMETRY_DIM,
        ]
        prev_action = inputs[
            ...,
            self.JOINT_DIM + self.GEOMETRY_DIM : self.JOINT_DIM
            + self.GEOMETRY_DIM
            + self.PREV_ACTION_DIM,
        ]
        time_ratio = inputs[..., -self.TIME_DIM :]
        return torch.cat(
            [
                self.joint_encoder(joints),
                self.geometry_encoder(geometry),
                self.prev_action_encoder(prev_action),
                self.time_encoder(time_ratio),
            ],
            dim=-1,
        )


class StructuredStateMLP(nn.Module):
    """State-only structured encoder followed by a fusion MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        activation = str(model_cfg.get("activation", "tanh"))
        fusion_hidden_sizes = model_cfg.get(
            "fusion_hidden_sizes",
            model_cfg.get("hidden_sizes", [256, 256]),
        )
        self.observation_encoder = StructuredObservationEncoder(input_dim, model_cfg)
        self.fusion = MLP(
            input_dim=self.observation_encoder.output_dim,
            output_dim=output_dim,
            hidden_sizes=fusion_hidden_sizes,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.observation_encoder(inputs))


class StructuredStateActionMLP(nn.Module):
    """Structured observation encoder plus a dedicated action branch."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_dim: int,
        model_cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.observation_dim = int(input_dim) - self.action_dim
        if self.observation_dim <= 0:
            raise ValueError("StructuredStateActionMLP requires observation_dim > 0.")

        activation = str(model_cfg.get("activation", "tanh"))
        fusion_hidden_sizes = model_cfg.get(
            "fusion_hidden_sizes",
            model_cfg.get("hidden_sizes", [256, 256]),
        )
        self.observation_encoder = StructuredObservationEncoder(
            self.observation_dim,
            model_cfg,
        )
        self.action_encoder, action_feature_dim = _build_branch(
            self.action_dim,
            model_cfg.get("action_hidden_sizes", [64, 64]),
            activation,
        )
        self.fusion = MLP(
            input_dim=self.observation_encoder.output_dim + action_feature_dim,
            output_dim=output_dim,
            hidden_sizes=fusion_hidden_sizes,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        observations = inputs[..., : self.observation_dim]
        actions = inputs[..., self.observation_dim :]
        fused_inputs = torch.cat(
            [
                self.observation_encoder(observations),
                self.action_encoder(actions),
            ],
            dim=-1,
        )
        return self.fusion(fused_inputs)


def build_state_network(
    *,
    input_dim: int,
    output_dim: int,
    model_cfg: dict[str, Any],
) -> nn.Module:
    model_type = str(model_cfg.get("type", "plain_mlp")).lower()
    activation = str(model_cfg.get("activation", "tanh"))
    hidden_sizes = model_cfg.get("hidden_sizes", [256, 256])

    if model_type == "plain_mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    if model_type == "structured_mlp":
        return StructuredStateMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            model_cfg=model_cfg,
        )
    raise ValueError(f"Unsupported model.type: {model_cfg.get('type')}")


def build_state_action_network(
    *,
    input_dim: int,
    output_dim: int,
    action_dim: int,
    model_cfg: dict[str, Any],
) -> nn.Module:
    model_type = str(model_cfg.get("type", "plain_mlp")).lower()
    activation = str(model_cfg.get("activation", "tanh"))
    hidden_sizes = model_cfg.get("hidden_sizes", [256, 256])

    if model_type == "plain_mlp":
        return MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
    if model_type == "structured_mlp":
        return StructuredStateActionMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            action_dim=action_dim,
            model_cfg=model_cfg,
        )
    raise ValueError(f"Unsupported model.type: {model_cfg.get('type')}")
