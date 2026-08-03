from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from hullwake.config import load_config  # noqa: E402
from hullwake.modeling import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_training_and_inference_forward_without_external_data_or_weights():
    config = load_config(
        ROOT / "configs" / "hullwake_r50_fpn.yaml",
        [
            "model.pretrained_backbone=false",
            "model.min_size=64",
            "model.max_size=64",
            "train.device=cpu",
            "data.workers=0",
        ],
    )
    model = build_model(config)
    image = torch.rand(3, 64, 64)
    hull_mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
    hull_mask[:, 20:32, 20:40] = 1
    wake_mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
    wake_mask[:, 22:30, 5:20] = 1
    target = {
        "boxes": torch.tensor([[20.0, 20.0, 40.0, 32.0]]),
        "labels": torch.tensor([1], dtype=torch.int64),
        "image_id": torch.tensor(1, dtype=torch.int64),
        "area": torch.tensor([240.0]),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
        "wake_attributes": torch.tensor([0], dtype=torch.int64),
        "wake_mask_valid": torch.tensor([True]),
        "masks": hull_mask,
        "wake_masks": wake_mask,
        "wake_like_mask": torch.zeros(1, 64, 64, dtype=torch.uint8),
        "water_clutter_mask": torch.zeros(1, 64, 64, dtype=torch.uint8),
        "source_id": torch.tensor(0, dtype=torch.int64),
    }
    model.train()
    losses = model([image], [target])
    expected = {
        "loss_classifier",
        "loss_box_reg",
        "loss_objectness",
        "loss_rpn_box_reg",
        "loss_wake",
        "loss_consistency",
        "loss_dominance",
        "loss_decorrelation",
    }
    assert expected.issubset(losses)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    assert model.roi_heads.final_classifier.weight.grad is not None
    assert model.roi_heads.wake_extractor.query.weight.grad is not None
    model.eval()
    with torch.inference_mode():
        output = model([image])[0]
    assert {"boxes", "labels", "scores", "attenuated_scores", "wake_drop"}.issubset(output)
