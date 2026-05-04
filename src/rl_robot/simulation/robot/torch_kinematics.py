#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Torch-based batched kinematics for GPU-friendly training environments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kinematics import RobotKinematics


def _rotation_x_batch(angle: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cos_a, -sin_a], dim=-1),
            torch.stack([zeros, sin_a, cos_a], dim=-1),
        ],
        dim=-2,
    )


def _rotation_z_batch(angle: torch.Tensor) -> torch.Tensor:
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    return torch.stack(
        [
            torch.stack([cos_a, -sin_a, zeros], dim=-1),
            torch.stack([sin_a, cos_a, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )


@dataclass
class TorchRobotKinematics:
    """Torch mirror of the analytical 4-DoF kinematics model."""

    joint_order: tuple[str, ...]
    joint_limits_rad: torch.Tensor
    zero_q_rad: torch.Tensor
    robot_position_world: torch.Tensor
    base_rotation: torch.Tensor
    yaw_to_shoulder: torch.Tensor
    shoulder_to_elbow: torch.Tensor
    elbow_to_wrist: torch.Tensor
    wrist_to_tool_tip: torch.Tensor

    @classmethod
    def from_robot_kinematics(
        cls,
        kinematics: RobotKinematics,
        device: torch.device,
    ) -> "TorchRobotKinematics":
        dtype = torch.float32
        robot_yaw = torch.tensor([kinematics.robot_yaw_rad], dtype=dtype, device=device)
        base_rotation = _rotation_z_batch(robot_yaw).squeeze(0)
        return cls(
            joint_order=kinematics.joint_order,
            joint_limits_rad=torch.as_tensor(
                kinematics.joint_limits_rad,
                dtype=dtype,
                device=device,
            ),
            zero_q_rad=torch.as_tensor(
                kinematics.zero_q_rad,
                dtype=dtype,
                device=device,
            ),
            robot_position_world=torch.as_tensor(
                kinematics.robot_position_world,
                dtype=dtype,
                device=device,
            ),
            base_rotation=base_rotation,
            yaw_to_shoulder=torch.as_tensor(
                kinematics.link_vectors_local["yaw_to_shoulder"],
                dtype=dtype,
                device=device,
            ),
            shoulder_to_elbow=torch.as_tensor(
                kinematics.link_vectors_local["shoulder_to_elbow"],
                dtype=dtype,
                device=device,
            ),
            elbow_to_wrist=torch.as_tensor(
                kinematics.link_vectors_local["elbow_to_wrist"],
                dtype=dtype,
                device=device,
            ),
            wrist_to_tool_tip=torch.as_tensor(
                kinematics.link_vectors_local["wrist_to_tool_tip"],
                dtype=dtype,
                device=device,
            ),
        )

    def clip_configuration(self, q_rad: torch.Tensor) -> torch.Tensor:
        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        return torch.clamp(q_rad, min=lower, max=upper)

    def normalize_configuration(self, q_rad: torch.Tensor) -> torch.Tensor:
        lower = self.joint_limits_rad[:, 0]
        upper = self.joint_limits_rad[:, 1]
        normalized = (q_rad - lower) / torch.clamp(upper - lower, min=1e-6)
        return 2.0 * normalized - 1.0

    def forward_kinematics(self, q_rad: torch.Tensor) -> torch.Tensor:
        if q_rad.ndim != 2 or q_rad.shape[1] != len(self.joint_order):
            raise ValueError("Expected q_rad shape [batch, dof].")

        q_yaw = q_rad[:, 0]
        q_shoulder = q_rad[:, 1]
        q_elbow = q_rad[:, 2]
        q_wrist = q_rad[:, 3]

        batch_size = q_rad.shape[0]
        base_rotation = self.base_rotation.unsqueeze(0).expand(batch_size, -1, -1)
        yaw_rotation = torch.bmm(base_rotation, _rotation_z_batch(q_yaw))
        shoulder_rotation = torch.bmm(yaw_rotation, _rotation_x_batch(-q_shoulder))
        elbow_rotation = torch.bmm(shoulder_rotation, _rotation_x_batch(-q_elbow))
        wrist_rotation = torch.bmm(elbow_rotation, _rotation_x_batch(-q_wrist))

        p_yaw = self.robot_position_world.unsqueeze(0).expand(batch_size, -1)
        p_shoulder = p_yaw + torch.einsum("bij,j->bi", yaw_rotation, self.yaw_to_shoulder)
        p_elbow = p_shoulder + torch.einsum(
            "bij,j->bi", shoulder_rotation, self.shoulder_to_elbow
        )
        p_wrist = p_elbow + torch.einsum("bij,j->bi", elbow_rotation, self.elbow_to_wrist)
        p_tool_tip = p_wrist + torch.einsum(
            "bij,j->bi", wrist_rotation, self.wrist_to_tool_tip
        )
        return p_tool_tip
