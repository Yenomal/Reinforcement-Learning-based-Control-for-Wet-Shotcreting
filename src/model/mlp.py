#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared MLP building blocks."""

from __future__ import annotations

from typing import Iterable

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
