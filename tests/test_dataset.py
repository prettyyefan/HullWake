import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")
from PIL import Image  # noqa: E402

from hullwake.data.curated_wake import CuratedWakeDataset  # noqa: E402


def test_extended_coco_loads_image_instance_and_diagnostic_masks(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (32, 24), color=(20, 40, 80)).save(image_root / "frame.png")
    payload = {
        "images": [
            {
                "id": 1,
                "file_name": "frame.png",
                "width": 32,
                "height": 24,
                "source": "smd",
                "sequence_id": "sequence_1",
            }
        ],
        "categories": [{"id": 7, "name": "vessel"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 8, 10, 6],
                "area": 60,
                "iscrowd": 0,
                "segmentation": [[10, 8, 20, 8, 20, 14, 10, 14]],
                "wake_attribute": "clear_wake",
                "wake_segmentation": [[10, 10, 3, 8, 2, 12, 9, 14]],
            }
        ],
        "regions": [
            {
                "id": 1,
                "image_id": 1,
                "type": "wake_like_negative",
                "segmentation": [[22, 2, 30, 2, 30, 5, 22, 5]],
            }
        ],
    }
    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(json.dumps(payload), encoding="utf-8")
    dataset = CuratedWakeDataset(image_root, annotation_file, require_wake_masks=True)
    image, target = dataset[0]
    assert image.shape == (3, 24, 32)
    assert image.dtype == torch.float32
    assert target["labels"].tolist() == [1]
    assert target["wake_attributes"].tolist() == [0]
    assert target["masks"].sum() > 0
    assert target["wake_masks"].sum() > 0
    assert target["wake_like_mask"].sum() > 0
