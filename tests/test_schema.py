from copy import deepcopy

from hullwake.data.schema import validate_coco_extension


def minimal_payload():
    return {
        "images": [
            {
                "id": 1,
                "file_name": "frame.jpg",
                "width": 64,
                "height": 48,
                "source": "smd",
                "sequence_id": "sequence_1",
            }
        ],
        "categories": [{"id": 1, "name": "vessel"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 8],
                "area": 160,
                "iscrowd": 0,
                "segmentation": [[10, 10, 30, 10, 30, 18, 10, 18]],
                "wake_attribute": "clear_wake",
                "wake_segmentation": [[10, 12, 4, 10, 2, 14, 8, 16]],
            }
        ],
        "regions": [],
    }


def test_valid_extended_coco_contract():
    issues = validate_coco_extension(minimal_payload(), require_wake_masks=True)
    assert not [issue for issue in issues if issue.level == "error"]


def test_clear_wake_requires_nonempty_mask():
    payload = deepcopy(minimal_payload())
    payload["annotations"][0]["wake_segmentation"] = []
    issues = validate_coco_extension(payload, require_wake_masks=True)
    assert any("clear_wake requires" in issue.message for issue in issues)


def test_path_traversal_and_bad_attribute_are_rejected():
    payload = deepcopy(minimal_payload())
    payload["images"][0]["file_name"] = "../private.jpg"
    payload["annotations"][0]["wake_attribute"] = "unknown"
    issues = validate_coco_extension(payload)
    messages = " ".join(issue.message for issue in issues)
    assert "safe relative path" in messages
    assert "wake_attribute" in messages
