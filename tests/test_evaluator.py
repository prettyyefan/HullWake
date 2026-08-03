import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from hullwake.evaluation import CocoWakeEvaluator  # noqa: E402


def test_perfect_predictions_have_high_group_ap_and_zero_drop():
    payload = {
        "info": {},
        "licenses": [],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64, "source": "smd"},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64, "source": "smd"},
        ],
        "categories": [{"id": 1, "name": "vessel"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 10],
                "area": 200,
                "iscrowd": 0,
                "wake_attribute": "clear_wake",
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [20, 20, 15, 12],
                "area": 180,
                "iscrowd": 0,
                "wake_attribute": "weak_no_wake",
            },
        ],
        "regions": [],
    }
    config = {
        "score_threshold": 0.5,
        "match_iou_threshold": 0.5,
        "wake_like_box_coverage_threshold": 0.25,
        "max_detections_per_image": 100,
    }
    predictions = {
        1: {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 20.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.99]),
            "attenuated_scores": torch.tensor([0.99]),
        },
        2: {
            "boxes": torch.tensor([[20.0, 20.0, 35.0, 32.0]]),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.98]),
            "attenuated_scores": torch.tensor([0.98]),
        },
    }
    result = CocoWakeEvaluator(payload, {1: 1}, config).evaluate(predictions)
    assert result["AP"] > 99.0
    assert result["AP_NoWake"] > 99.0
    assert result["WG_AP"] > 99.0
    assert result["FP_WakeLike"] == 0
    assert abs(result["Delta_wake"]) < 1e-8
