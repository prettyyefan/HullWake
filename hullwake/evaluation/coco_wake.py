from __future__ import annotations

import contextlib
import copy
import io
from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from hullwake.data.masks import decode_segmentation


def predictions_to_coco(
    predictions: dict[int, dict[str, torch.Tensor]], label_to_category_id: dict[int, int]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for image_id in sorted(predictions):
        output = predictions[image_id]
        boxes = output["boxes"].detach().cpu()
        labels = output["labels"].detach().cpu()
        scores = output["scores"].detach().cpu()
        attenuated = output.get("attenuated_scores", scores).detach().cpu()
        for box, label, score, attenuated_score in zip(boxes, labels, scores, attenuated):
            x1, y1, x2, y2 = map(float, box.tolist())
            records.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(label_to_category_id[int(label)]),
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": float(score),
                    "attenuated_score": float(attenuated_score),
                    "wake_drop": float(score - attenuated_score),
                }
            )
    return records


def _coco(payload: dict[str, Any]) -> COCO:
    data = copy.deepcopy(payload)
    data.setdefault("info", {})
    data.setdefault("licenses", [])
    api = COCO()
    api.dataset = data
    api.createIndex()
    return api


def _run_coco(
    payload: dict[str, Any],
    detections: list[dict[str, Any]],
    *,
    image_ids: Iterable[int] | None,
    max_detections: int,
) -> dict[str, float | None]:
    selected_ids = (
        sorted(int(value) for value in image_ids)
        if image_ids is not None
        else sorted(int(image["id"]) for image in payload["images"])
    )
    selected_set = set(selected_ids)
    ground_truth_count = sum(
        int(annotation.get("iscrowd", 0)) == 0
        and int(annotation["image_id"]) in selected_set
        for annotation in payload["annotations"]
    )
    if ground_truth_count == 0:
        return {"AP": None, "AP50": None, "AP75": None}
    filtered_detections = [
        {key: value for key, value in detection.items() if key != "attenuated_score" and key != "wake_drop"}
        for detection in detections
        if int(detection["image_id"]) in selected_set
    ]
    ground_truth = _coco(payload)
    if not filtered_detections:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        predicted = ground_truth.loadRes(filtered_detections)
        evaluator = COCOeval(ground_truth, predicted, iouType="bbox")
        evaluator.params.imgIds = selected_ids
        evaluator.params.maxDets = [1, 10, int(max_detections)]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    values = evaluator.stats

    def value(index: int) -> float | None:
        return None if values[index] < 0 else float(values[index] * 100.0)

    return {"AP": value(0), "AP50": value(1), "AP75": value(2)}


def _group_payload(payload: dict[str, Any], attribute: str) -> dict[str, Any]:
    grouped = copy.deepcopy(payload)
    for annotation in grouped["annotations"]:
        in_group = annotation.get("wake_attribute") == attribute
        annotation["ignore"] = 0 if in_group else 1
        annotation["iscrowd"] = 0 if in_group else 1
    return grouped


def _box_iou_xywh(first: list[float], second: list[float]) -> float:
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


class CocoWakeEvaluator:
    def __init__(
        self,
        coco_payload: dict[str, Any],
        label_to_category_id: dict[int, int],
        evaluation_config: dict[str, Any],
    ) -> None:
        self.payload = copy.deepcopy(coco_payload)
        self.label_to_category_id = dict(label_to_category_id)
        self.config = dict(evaluation_config)
        self.images = {int(image["id"]): image for image in self.payload["images"]}
        self.annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.regions_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.payload["annotations"]:
            self.annotations_by_image[int(annotation["image_id"])].append(annotation)
        for region in self.payload.get("regions", []):
            self.regions_by_image[int(region["image_id"])].append(region)

    def _wake_like_false_positives(
        self, detections: list[dict[str, Any]], selected_ids: set[int]
    ) -> int:
        score_threshold = float(self.config["score_threshold"])
        match_threshold = float(self.config["match_iou_threshold"])
        coverage_threshold = float(self.config["wake_like_box_coverage_threshold"])
        count = 0
        masks: dict[int, np.ndarray] = {}
        for detection in detections:
            image_id = int(detection["image_id"])
            if image_id not in selected_ids or float(detection["score"]) < score_threshold:
                continue
            true_boxes = [annotation["bbox"] for annotation in self.annotations_by_image[image_id]]
            if any(_box_iou_xywh(detection["bbox"], box) >= match_threshold for box in true_boxes):
                continue
            if image_id not in masks:
                image = self.images[image_id]
                union = np.zeros((int(image["height"]), int(image["width"])), dtype=np.uint8)
                for region in self.regions_by_image[image_id]:
                    if region.get("type") == "wake_like_negative":
                        union |= decode_segmentation(
                            region["segmentation"], union.shape[0], union.shape[1]
                        )
                masks[image_id] = union
            x, y, width, height = map(float, detection["bbox"])
            image_mask = masks[image_id]
            x1 = max(0, min(image_mask.shape[1], int(np.floor(x))))
            y1 = max(0, min(image_mask.shape[0], int(np.floor(y))))
            x2 = max(0, min(image_mask.shape[1], int(np.ceil(x + width))))
            y2 = max(0, min(image_mask.shape[0], int(np.ceil(y + height))))
            coverage = float(image_mask[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 0.0
            if coverage >= coverage_threshold:
                count += 1
        return count

    def _wake_drop(self, detections: list[dict[str, Any]], selected_ids: set[int]) -> float | None:
        threshold = float(self.config["match_iou_threshold"])
        drops: list[float] = []
        detections_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for detection in detections:
            if int(detection["image_id"]) in selected_ids:
                detections_by_image[int(detection["image_id"])].append(detection)
        for image_id in selected_ids:
            ground_truths = self.annotations_by_image[image_id]
            used_ground_truth: set[int] = set()
            ordered = sorted(
                detections_by_image.get(image_id, []), key=lambda item: float(item["score"]), reverse=True
            )
            for detection in ordered:
                best_index = None
                best_iou = threshold
                for index, annotation in enumerate(ground_truths):
                    if index in used_ground_truth:
                        continue
                    if int(annotation["category_id"]) != int(detection["category_id"]):
                        continue
                    overlap = _box_iou_xywh(detection["bbox"], annotation["bbox"])
                    if overlap >= best_iou:
                        best_iou = overlap
                        best_index = index
                if best_index is not None:
                    used_ground_truth.add(best_index)
                    drops.append(
                        float(detection["score"]) - float(detection.get("attenuated_score", detection["score"]))
                    )
        return float(np.mean(drops)) if drops else None

    def _subset_metrics(
        self, detections: list[dict[str, Any]], selected_ids: set[int]
    ) -> dict[str, Any]:
        maximum = int(self.config["max_detections_per_image"])
        clear = _run_coco(
            _group_payload(self.payload, "clear_wake"),
            detections,
            image_ids=selected_ids,
            max_detections=maximum,
        )["AP"]
        no_wake = _run_coco(
            _group_payload(self.payload, "weak_no_wake"),
            detections,
            image_ids=selected_ids,
            max_detections=maximum,
        )["AP"]
        valid_groups = [value for value in (clear, no_wake) if value is not None]
        worst_group = min(valid_groups) if len(valid_groups) == 2 else None
        return {
            "AP_ClearWake": clear,
            "AP_NoWake": no_wake,
            "WG_AP": worst_group,
            "FP_WakeLike": self._wake_like_false_positives(detections, selected_ids),
            "Delta_wake": self._wake_drop(detections, selected_ids),
        }

    def evaluate(self, predictions: dict[int, dict[str, torch.Tensor]]) -> dict[str, Any]:
        detections = predictions_to_coco(predictions, self.label_to_category_id)
        image_ids = set(self.images)
        maximum = int(self.config["max_detections_per_image"])
        standard = _run_coco(
            self.payload, detections, image_ids=image_ids, max_detections=maximum
        )
        result: dict[str, Any] = {
            "AP": standard["AP"],
            "AP50_95": standard["AP"],
            "AP50": standard["AP50"],
            "AP75": standard["AP75"],
            **self._subset_metrics(detections, image_ids),
        }
        sources = sorted({str(image["source"]) for image in self.images.values()})
        result["by_source"] = {}
        for source in sources:
            source_ids = {
                image_id
                for image_id, image in self.images.items()
                if str(image["source"]) == source
            }
            result["by_source"][source] = self._subset_metrics(detections, source_ids)
        result["number_of_images"] = len(image_ids)
        result["number_of_detections"] = len(detections)
        result["protocol"] = {
            "score_threshold": float(self.config["score_threshold"]),
            "match_iou_threshold": float(self.config["match_iou_threshold"]),
            "wake_like_box_coverage_threshold": float(
                self.config["wake_like_box_coverage_threshold"]
            ),
        }
        return result
