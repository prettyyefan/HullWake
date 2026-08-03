from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class WakeExtractorOutput:
    descriptor: torch.Tensor
    response_logits: torch.Tensor
    attenuation_map: torch.Tensor
    points: torch.Tensor
    valid_points: torch.Tensor
    direction_gate: torch.Tensor
    orientation: torch.Tensor
    length: torch.Tensor
    width: torch.Tensor
    batch_indices: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class ProposalAlignedWakeExtractor(nn.Module):
    """Bidirectional proposal-anchored corridor sampler with directional attention."""

    def __init__(
        self,
        *,
        input_channels: int,
        embedding_dim: int,
        grid_size: int,
        length_max_ratio: float,
        width_max_ratio: float,
        fpn_strides: Sequence[int],
        epsilon: float,
        parallel_bias_init: float = 1.0,
        perpendicular_bias_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.embedding_dim = int(embedding_dim)
        self.grid_size = int(grid_size)
        self.points_per_direction = self.grid_size * self.grid_size
        self.length_max_ratio = float(length_max_ratio)
        self.width_max_ratio = float(width_max_ratio)
        self.fpn_strides = tuple(int(value) for value in fpn_strides)
        self.epsilon = float(epsilon)

        self.orientation_head = nn.Linear(embedding_dim, 1)
        self.length_head = nn.Linear(embedding_dim, 1)
        self.width_head = nn.Linear(embedding_dim, 1)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(input_channels, embedding_dim)
        self.value = nn.Linear(input_channels, embedding_dim)
        self.response = nn.Linear(input_channels, 1)
        self.direction_gate = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.raw_eta_parallel = nn.Parameter(
            torch.tensor(_inverse_softplus(float(parallel_bias_init)), dtype=torch.float32)
        )
        self.raw_eta_perpendicular = nn.Parameter(
            torch.tensor(_inverse_softplus(float(perpendicular_bias_init)), dtype=torch.float32)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

    @staticmethod
    def flatten_proposals(proposals: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not proposals:
            raise ValueError("proposals must contain at least one image")
        device = proposals[0].device
        boxes = torch.cat(proposals, dim=0)
        image_indices = torch.cat(
            [
                torch.full((len(boxes_i),), index, dtype=torch.long, device=device)
                for index, boxes_i in enumerate(proposals)
            ],
            dim=0,
        )
        return boxes, image_indices

    def _assign_levels(self, boxes: torch.Tensor, number_of_levels: int) -> torch.Tensor:
        widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(self.epsilon)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(self.epsilon)
        scale = torch.sqrt(widths * heights)
        # Standard FPN rule: canonical scale 224 maps to P4. Indices 0..3 represent P2..P5.
        target = torch.floor(4.0 + torch.log2(scale / 224.0 + self.epsilon)) - 2.0
        return target.to(torch.long).clamp_(min=0, max=number_of_levels - 1)

    def _make_points(
        self, boxes: torch.Tensor, hull_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = torch.stack(
            ((boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5),
            dim=-1,
        )
        side_scale = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        orientation = math.pi * torch.tanh(self.orientation_head(hull_features).squeeze(-1))
        length = (
            side_scale
            * torch.sigmoid(self.length_head(hull_features).squeeze(-1))
            * self.length_max_ratio
        )
        width = (
            side_scale
            * torch.sigmoid(self.width_head(hull_features).squeeze(-1))
            * self.width_max_ratio
        )
        parallel = torch.stack((torch.cos(orientation), torch.sin(orientation)), dim=-1)
        perpendicular = torch.stack((-torch.sin(orientation), torch.cos(orientation)), dim=-1)

        alpha_fraction = torch.linspace(
            0.0, 1.0, self.grid_size, device=boxes.device, dtype=boxes.dtype
        )
        beta_fraction = torch.linspace(
            -0.5, 0.5, self.grid_size, device=boxes.device, dtype=boxes.dtype
        )
        alpha_grid, beta_grid = torch.meshgrid(
            alpha_fraction, beta_fraction, indexing="ij"
        )
        alpha = alpha_grid.reshape(1, -1, 1) * length[:, None, None]
        beta = beta_grid.reshape(1, -1, 1) * width[:, None, None]
        lateral = beta * perpendicular[:, None, :]
        plus = centers[:, None, :] - alpha * parallel[:, None, :] + lateral
        minus = centers[:, None, :] + alpha * parallel[:, None, :] + lateral
        points = torch.stack((plus, minus), dim=1)

        eta_parallel = F.softplus(self.raw_eta_parallel)
        eta_perpendicular = F.softplus(self.raw_eta_perpendicular)
        position_bias = (
            -eta_perpendicular * torch.abs(beta_grid.reshape(-1))
            + eta_parallel * alpha_grid.reshape(-1)
        )
        return points, position_bias, orientation, length, width

    def _sample_features(
        self,
        feature_maps: list[torch.Tensor],
        points: torch.Tensor,
        batch_indices: torch.Tensor,
        levels: torch.Tensor,
        image_shapes: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proposal_count, directions, point_count, _ = points.shape
        sampled = points.new_zeros(
            (proposal_count, directions, point_count, self.input_channels)
        )
        valid = torch.zeros(
            (proposal_count, directions, point_count), dtype=torch.bool, device=points.device
        )
        image_heights = points.new_tensor([shape[0] for shape in image_shapes])
        image_widths = points.new_tensor([shape[1] for shape in image_shapes])
        valid = (
            (points[..., 0] >= 0)
            & (points[..., 1] >= 0)
            & (points[..., 0] <= image_widths[batch_indices, None, None] - 1)
            & (points[..., 1] <= image_heights[batch_indices, None, None] - 1)
        )

        for level_index, (feature, stride) in enumerate(zip(feature_maps, self.fpn_strides)):
            selected = torch.where(levels == level_index)[0]
            if selected.numel() == 0:
                continue
            level_points = points[selected]
            feature_height, feature_width = feature.shape[-2:]
            x_feature = level_points[..., 0] / float(stride)
            y_feature = level_points[..., 1] / float(stride)
            if feature_width > 1:
                x_normalized = 2.0 * x_feature / float(feature_width - 1) - 1.0
            else:
                x_normalized = torch.zeros_like(x_feature)
            if feature_height > 1:
                y_normalized = 2.0 * y_feature / float(feature_height - 1) - 1.0
            else:
                y_normalized = torch.zeros_like(y_feature)
            grid = torch.stack((x_normalized, y_normalized), dim=-1)
            source = feature[batch_indices[selected]]
            values = F.grid_sample(
                source,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            sampled[selected] = values.permute(0, 2, 3, 1)
        return sampled, valid

    def forward(
        self,
        features: list[torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        hull_features: torch.Tensor,
    ) -> WakeExtractorOutput:
        boxes, batch_indices = self.flatten_proposals(proposals)
        if boxes.shape[0] != hull_features.shape[0]:
            raise ValueError("Hull feature count does not match proposal count")
        if len(features) != len(self.fpn_strides):
            raise ValueError(
                f"Expected {len(self.fpn_strides)} FPN maps, received {len(features)}"
            )
        points, position_bias, orientation, length, width = self._make_points(
            boxes, hull_features
        )
        levels = self._assign_levels(boxes, len(features))
        sampled_features, valid_points = self._sample_features(
            features, points, batch_indices, levels, image_shapes
        )

        query = self.query(hull_features)[:, None, None, :]
        keys = self.key(sampled_features)
        values = self.value(sampled_features)
        attention_logits = torch.sum(query * keys, dim=-1) / math.sqrt(self.embedding_dim)
        attention_logits = attention_logits + position_bias[None, None, :]
        attention_logits = attention_logits.masked_fill(~valid_points, -1e4)
        attention = torch.softmax(attention_logits, dim=-1)
        directional_descriptors = torch.sum(attention[..., None] * values, dim=2)
        plus_descriptor = directional_descriptors[:, 0]
        minus_descriptor = directional_descriptors[:, 1]
        direction_gate = torch.sigmoid(
            self.direction_gate(
                torch.cat((plus_descriptor, minus_descriptor, hull_features), dim=-1)
            )
        )
        descriptor = (
            direction_gate * plus_descriptor + (1.0 - direction_gate) * minus_descriptor
        )
        descriptor = self.output_norm(descriptor)
        response_logits = self.response(sampled_features).squeeze(-1)
        proposal_count = response_logits.shape[0]
        plus_map = torch.sigmoid(response_logits[:, 0]).reshape(
            proposal_count, self.grid_size, self.grid_size
        )
        minus_map = torch.sigmoid(response_logits[:, 1]).reshape(
            proposal_count, self.grid_size, self.grid_size
        )
        attenuation_map = (
            direction_gate[:, :, None] * plus_map
            + (1.0 - direction_gate[:, :, None]) * minus_map
        )
        return WakeExtractorOutput(
            descriptor=descriptor,
            response_logits=response_logits,
            attenuation_map=attenuation_map,
            points=points,
            valid_points=valid_points,
            direction_gate=direction_gate.squeeze(-1),
            orientation=orientation,
            length=length,
            width=width,
            batch_indices=batch_indices,
        )
