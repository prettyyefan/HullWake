from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from hullwake.modeling.roi_heads import HullWakeRoIHeads


class HullWakeTransform(GeneralizedRCNNTransform):
    """TorchVision transform that also resizes HullWake-specific masks."""

    def resize(
        self, image: torch.Tensor, target: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        image, target = super().resize(image, target)
        if target is None:
            return image, target
        output = dict(target)
        new_size = image.shape[-2:]
        for key in ("wake_masks", "wake_like_mask", "water_clutter_mask"):
            if key not in output:
                continue
            masks = output[key]
            if masks.numel() == 0:
                output[key] = masks.new_zeros((masks.shape[0], *new_size))
            else:
                dtype = masks.dtype
                output[key] = F.interpolate(
                    masks[:, None].float(), size=new_size, mode="nearest"
                )[:, 0].to(dtype=dtype)
        return image, output


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_config = config["model"]
    data_config = config["data"]
    if model_config["backbone"] != "resnet50_fpn":
        raise ValueError("This release implements the paper backbone: resnet50_fpn")
    backbone_weights = (
        ResNet50_Weights.IMAGENET1K_V1 if bool(model_config["pretrained_backbone"]) else None
    )
    detector = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=int(data_config["num_classes"]),
        min_size=int(model_config["min_size"]),
        max_size=int(model_config["max_size"]),
        box_score_thresh=float(model_config["score_threshold"]),
        box_nms_thresh=float(model_config["nms_threshold"]),
        box_detections_per_img=int(model_config["detections_per_image"]),
    )
    base_roi_heads = detector.roi_heads
    detector.roi_heads = HullWakeRoIHeads(
        base=base_roi_heads,
        feature_channels=int(detector.backbone.out_channels),
        embedding_dim=int(model_config["embedding_dim"]),
        fpn_level_indices=list(model_config["fpn_levels"]),
        fpn_strides=list(model_config["fpn_strides"]),
        grid_size=int(model_config["corridor_grid_size"]),
        length_max_ratio=float(model_config["length_max_ratio"]),
        width_max_ratio=float(model_config["width_max_ratio"]),
        directional_bias_parallel_init=float(
            model_config["directional_bias_parallel_init"]
        ),
        directional_bias_perpendicular_init=float(
            model_config["directional_bias_perpendicular_init"]
        ),
        use_wake_extractor=bool(model_config["use_wake_extractor"]),
        use_wake_fusion=bool(model_config["use_wake_fusion"]),
        detach_wake_in_consistency=bool(model_config["detach_wake_in_consistency"]),
        loss_config=config["loss"],
    )
    old_transform = detector.transform
    detector.transform = HullWakeTransform(
        min_size=int(model_config["min_size"]),
        max_size=int(model_config["max_size"]),
        image_mean=old_transform.image_mean,
        image_std=old_transform.image_std,
        size_divisible=old_transform.size_divisible,
        fixed_size=old_transform.fixed_size,
    )
    return detector
