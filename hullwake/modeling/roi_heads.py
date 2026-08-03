from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops

from hullwake.modeling.losses import (
    hull_wake_decorrelation_loss,
    wake_consistency_loss,
    wake_dominance_loss,
    wake_response_loss,
)
from hullwake.modeling.wake_extractor import ProposalAlignedWakeExtractor


class HullWakeRoIHeads(RoIHeads):
    def __init__(
        self,
        *,
        base: RoIHeads,
        feature_channels: int,
        embedding_dim: int,
        fpn_level_indices: list[int],
        fpn_strides: list[int],
        grid_size: int,
        length_max_ratio: float,
        width_max_ratio: float,
        directional_bias_parallel_init: float,
        directional_bias_perpendicular_init: float,
        use_wake_extractor: bool,
        use_wake_fusion: bool,
        detach_wake_in_consistency: bool,
        loss_config: dict[str, Any],
    ) -> None:
        super().__init__(
            base.box_roi_pool,
            base.box_head,
            base.box_predictor,
            base.proposal_matcher.high_threshold,
            base.proposal_matcher.low_threshold,
            base.fg_bg_sampler.batch_size_per_image,
            base.fg_bg_sampler.positive_fraction,
            base.box_coder.weights,
            base.score_thresh,
            base.nms_thresh,
            base.detections_per_img,
        )
        self.use_wake_extractor = bool(use_wake_extractor)
        self.use_wake_fusion = bool(use_wake_fusion)
        self.detach_wake_in_consistency = bool(detach_wake_in_consistency)
        self.loss_config = dict(loss_config)
        self.fpn_level_indices = tuple(int(index) for index in fpn_level_indices)
        representation_size = int(base.box_predictor.cls_score.in_features)
        number_of_classes = int(base.box_predictor.cls_score.out_features)
        self.hull_projection = nn.Sequential(
            nn.Linear(representation_size, embedding_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(embedding_dim),
        )
        self.wake_extractor = ProposalAlignedWakeExtractor(
            input_channels=feature_channels,
            embedding_dim=embedding_dim,
            grid_size=grid_size,
            length_max_ratio=length_max_ratio,
            width_max_ratio=width_max_ratio,
            fpn_strides=fpn_strides,
            epsilon=float(loss_config["epsilon"]),
            parallel_bias_init=directional_bias_parallel_init,
            perpendicular_bias_init=directional_bias_perpendicular_init,
        )
        self.wake_projection = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1),
        )
        self.final_classifier = nn.Linear(embedding_dim, number_of_classes)
        self.hull_classifier = nn.Linear(embedding_dim, number_of_classes)
        self.wake_classifier = nn.Linear(embedding_dim, number_of_classes)
        self._reset_new_parameters()

    def _reset_new_parameters(self) -> None:
        for module in (
            self.hull_projection,
            self.wake_projection,
            self.fusion_gate,
            self.final_classifier,
            self.hull_classifier,
            self.wake_classifier,
        ):
            for layer in module.modules() if isinstance(module, nn.Module) else ():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    @staticmethod
    def _sample_masks_at_points(masks: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        """Sample one full-resolution mask per proposal at [N, 2, K, 2] image points."""
        count = points.shape[0]
        if count == 0:
            return points.new_zeros(points.shape[:-1])
        masks = masks.to(device=points.device, dtype=points.dtype)
        height, width = masks.shape[-2:]
        x = points[..., 0]
        y = points[..., 1]
        x_normalized = (
            2.0 * x / float(width - 1) - 1.0 if width > 1 else torch.zeros_like(x)
        )
        y_normalized = (
            2.0 * y / float(height - 1) - 1.0 if height > 1 else torch.zeros_like(y)
        )
        grid = torch.stack((x_normalized, y_normalized), dim=-1)
        sampled = F.grid_sample(
            masks[:, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[:, 0]

    def _wake_training_targets(
        self,
        *,
        points: torch.Tensor,
        proposals: list[torch.Tensor],
        matched_indices: list[torch.Tensor],
        labels: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wake_targets = points.new_zeros(points.shape[:-1])
        valid_proposals = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        negative_weights = points.new_ones((points.shape[0],))
        offset = 0
        for proposal_boxes, matched, image_labels, target in zip(
            proposals, matched_indices, labels, targets
        ):
            count = len(proposal_boxes)
            if count == 0:
                continue
            selected_points = points[offset : offset + count]
            positive = image_labels > 0
            negative = image_labels == 0
            if torch.any(positive):
                matched_positive = matched[positive]
                source_masks = target["wake_masks"][matched_positive]
                wake_targets[offset : offset + count][positive] = self._sample_masks_at_points(
                    source_masks, selected_points[positive]
                )
                if "wake_mask_valid" in target:
                    valid_proposals[offset : offset + count][positive] = target[
                        "wake_mask_valid"
                    ][matched_positive]
                else:
                    valid_proposals[offset : offset + count][positive] = True
            valid_proposals[offset : offset + count][negative] = True

            if torch.any(negative):
                hard_union = torch.logical_or(
                    target["wake_like_mask"].bool(), target["water_clutter_mask"].bool()
                ).any(dim=0, keepdim=True)
                repeated = hard_union.expand(int(negative.sum().item()), -1, -1)
                sampled_hard = self._sample_masks_at_points(
                    repeated, selected_points[negative]
                )
                hard_fraction = sampled_hard.mean(dim=(1, 2))
                threshold = float(self.loss_config["hard_negative_point_fraction"])
                extra = float(self.loss_config["hard_negative_extra_weight"])
                negative_weights[offset : offset + count][negative] = 1.0 + extra * (
                    hard_fraction >= threshold
                ).to(points.dtype)
            offset += count
        return wake_targets, valid_proposals, negative_weights

    def _postprocess_with_attenuation(
        self,
        class_logits: torch.Tensor,
        attenuated_class_logits: torch.Tensor,
        box_regression: torch.Tensor,
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        device = class_logits.device
        number_of_classes = class_logits.shape[-1]
        boxes_per_image = [boxes.shape[0] for boxes in proposals]
        predicted_boxes = self.box_coder.decode(box_regression, proposals)
        predicted_scores = F.softmax(class_logits, dim=-1)
        attenuated_scores = F.softmax(attenuated_class_logits, dim=-1)
        boxes_list = predicted_boxes.split(boxes_per_image, 0)
        scores_list = predicted_scores.split(boxes_per_image, 0)
        attenuated_list = attenuated_scores.split(boxes_per_image, 0)

        results: list[dict[str, torch.Tensor]] = []
        for boxes, scores, atten_scores, image_shape in zip(
            boxes_list, scores_list, attenuated_list, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            labels = torch.arange(number_of_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            boxes = boxes[:, 1:].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            atten_scores = atten_scores[:, 1:].reshape(-1)
            labels = labels[:, 1:].reshape(-1)

            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, atten_scores, labels = (
                boxes[keep],
                scores[keep],
                atten_scores[keep],
                labels[keep],
            )
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, atten_scores, labels = (
                boxes[keep],
                scores[keep],
                atten_scores[keep],
                labels[keep],
            )
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            scores = scores[keep]
            atten_scores = atten_scores[keep]
            results.append(
                {
                    "boxes": boxes[keep],
                    "labels": labels[keep],
                    "scores": scores,
                    "attenuated_scores": atten_scores,
                    "wake_drop": scores - atten_scores,
                }
            )
        return results

    def forward(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        if targets is not None:
            for target in targets:
                if target["boxes"].dtype not in (torch.float16, torch.float32, torch.float64):
                    raise TypeError("target boxes must be floating point")
                if target["labels"].dtype != torch.int64:
                    raise TypeError("target labels must be int64")

        if self.training:
            proposals, matched_indices, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            matched_indices = None
            labels = None
            regression_targets = None

        roi_features = self.box_roi_pool(features, proposals, image_shapes)
        box_representation = self.box_head(roi_features)
        base_class_logits, box_regression = self.box_predictor(box_representation)
        losses: dict[str, torch.Tensor] = {}

        if not self.use_wake_extractor:
            class_logits = base_class_logits
            attenuated_class_logits = base_class_logits
        else:
            hull_features = self.hull_projection(box_representation)
            feature_values = list(features.values())
            selected_fpn = [feature_values[index] for index in self.fpn_level_indices]
            wake_output = self.wake_extractor(
                selected_fpn, proposals, image_shapes, hull_features
            )
            projected_wake = self.wake_projection(wake_output.descriptor)
            fusion_gate = torch.sigmoid(
                self.fusion_gate(torch.cat((hull_features, wake_output.descriptor), dim=-1))
            )
            fused_features = (
                hull_features + fusion_gate * projected_wake
                if self.use_wake_fusion
                else hull_features
            )
            class_logits = self.final_classifier(fused_features)

            resized_response = F.interpolate(
                wake_output.attenuation_map[:, None],
                size=roi_features.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            attenuated_roi = roi_features * (
                1.0 - float(self.loss_config["lambda_att"]) * resized_response
            )
            attenuated_representation = self.box_head(attenuated_roi)
            attenuated_hull = self.hull_projection(attenuated_representation)
            if self.detach_wake_in_consistency:
                consistency_gate = fusion_gate.detach()
                consistency_wake = wake_output.descriptor.detach()
            else:
                consistency_gate = fusion_gate
                consistency_wake = wake_output.descriptor
            attenuated_fused = (
                attenuated_hull + consistency_gate * self.wake_projection(consistency_wake)
                if self.use_wake_fusion
                else attenuated_hull
            )
            attenuated_class_logits = self.final_classifier(attenuated_fused)

            if self.training:
                if targets is None or matched_indices is None or labels is None:
                    raise RuntimeError("Training HullWake requires targets and matched proposals")
                flat_labels = torch.cat(labels, dim=0)
                wake_targets, wake_valid, negative_weights = self._wake_training_targets(
                    points=wake_output.points,
                    proposals=proposals,
                    matched_indices=matched_indices,
                    labels=labels,
                    targets=targets,
                )
                if bool(self.loss_config["enable_wake_supervision"]):
                    losses["loss_wake"] = float(self.loss_config["beta_wake"]) * wake_response_loss(
                        wake_output.response_logits, wake_targets, wake_valid
                    )
                if bool(self.loss_config["enable_consistency"]):
                    losses["loss_consistency"] = float(
                        self.loss_config["beta_cons"]
                    ) * wake_consistency_loss(
                        class_logits, attenuated_class_logits, flat_labels > 0
                    )
                if bool(self.loss_config["enable_dominance"]):
                    hull_logits = self.hull_classifier(hull_features)
                    wake_logits = self.wake_classifier(wake_output.descriptor)
                    losses["loss_dominance"] = float(
                        self.loss_config["beta_dom"]
                    ) * wake_dominance_loss(
                        hull_logits,
                        wake_logits,
                        flat_labels,
                        margin=float(self.loss_config["dominance_margin"]),
                        lambda_negative=float(self.loss_config["lambda_neg"]),
                        negative_weights=negative_weights,
                    )
                if bool(self.loss_config["enable_decorrelation"]):
                    losses["loss_decorrelation"] = float(
                        self.loss_config["beta_dec"]
                    ) * hull_wake_decorrelation_loss(
                        hull_features,
                        wake_output.descriptor,
                        float(self.loss_config["epsilon"]),
                    )

        result: list[dict[str, torch.Tensor]] = []
        if self.training:
            if labels is None or regression_targets is None:
                raise RuntimeError("Missing training targets")
            loss_classifier, loss_box_regression = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses["loss_classifier"] = loss_classifier
            losses["loss_box_reg"] = loss_box_regression
        else:
            result = self._postprocess_with_attenuation(
                class_logits,
                attenuated_class_logits,
                box_regression,
                proposals,
                image_shapes,
            )
        return result, losses
