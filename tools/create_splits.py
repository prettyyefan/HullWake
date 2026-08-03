#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hullwake.data.schema import validate_coco_extension
from hullwake.utils.io import sha256_file

SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create source-stratified, sequence-safe Curated-Wake splits"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--prefix", default="curated_wake")
    return parser.parse_args()


def _source_seed(seed: int, source: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def assign_groups(
    images: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int
) -> dict[str, set[int]]:
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for image in images:
        by_source[str(image["source"])][str(image["sequence_id"])].append(image)
    assignments = {name: set() for name in SPLIT_NAMES}
    for source in sorted(by_source):
        groups = list(by_source[source].values())
        random.Random(_source_seed(seed, source)).shuffle(groups)
        total = sum(len(group) for group in groups)
        targets = [ratio * total for ratio in ratios]
        current = [0, 0, 0]
        # Larger sequences first after the seeded shuffle keeps leakage safety while reducing ratio error.
        groups.sort(key=len, reverse=True)
        for group in groups:
            deficits = [targets[index] - current[index] for index in range(3)]
            split_index = max(range(3), key=lambda index: (deficits[index], -current[index], -index))
            current[split_index] += len(group)
            assignments[SPLIT_NAMES[split_index]].update(int(image["id"]) for image in group)
    return assignments


def subset_payload(
    payload: dict[str, Any], image_ids: set[int], split: str, seed: int, master_hash: str | None
) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(value)
        for key, value in payload.items()
        if key not in ("images", "annotations", "regions")
    }
    result["images"] = [
        copy.deepcopy(image) for image in payload["images"] if int(image["id"]) in image_ids
    ]
    result["annotations"] = [
        copy.deepcopy(annotation)
        for annotation in payload["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    result["regions"] = [
        copy.deepcopy(region)
        for region in payload["regions"]
        if int(region["image_id"]) in image_ids
    ]
    result.setdefault("info", {})
    result["info"].update(
        {
            "hullwake_split": split,
            "hullwake_split_seed": seed,
            "master_annotation_sha256": master_hash,
        }
    )
    return result


def main() -> int:
    args = parse_args()
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(value <= 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise SystemExit("Split ratios must be positive and sum to 1.0")
    source_path = Path(args.annotations)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    issues = validate_coco_extension(payload)
    errors = [str(issue) for issue in issues if issue.level == "error"]
    if errors:
        raise SystemExit("Master annotation is invalid:\n" + "\n".join(errors[:20]))
    assignments = assign_groups(payload["images"], ratios, args.seed)
    all_ids = set().union(*assignments.values())
    expected_ids = {int(image["id"]) for image in payload["images"]}
    if all_ids != expected_ids or sum(len(value) for value in assignments.values()) != len(all_ids):
        raise RuntimeError("Internal split assignment error: missing or duplicated image IDs")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    master_hash = sha256_file(source_path)
    for split in SPLIT_NAMES:
        split_payload = subset_payload(payload, assignments[split], split, args.seed, master_hash)
        destination = output / f"{args.prefix}_{split}.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(split_payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        sources = Counter(image["source"] for image in split_payload["images"])
        attributes = Counter(
            annotation["wake_attribute"] for annotation in split_payload["annotations"]
        )
        print(
            f"{split}: images={len(split_payload['images'])} "
            f"instances={len(split_payload['annotations'])} "
            f"sources={dict(sorted(sources.items()))} "
            f"wake_attributes={dict(sorted(attributes.items()))} -> {destination}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
