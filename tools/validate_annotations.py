#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from hullwake.data.schema import validate_coco_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extended COCO Curated-Wake JSON")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--images")
    parser.add_argument("--require-files", action="store_true")
    parser.add_argument("--require-wake-masks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_files and not args.images:
        raise SystemExit("--require-files also requires --images")
    with Path(args.annotations).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    issues = validate_coco_extension(
        payload,
        images_root=args.images,
        require_files=args.require_files,
        require_wake_masks=args.require_wake_masks,
    )
    for issue in issues:
        print(issue)
    counts = Counter(issue.level for issue in issues)
    print(
        f"images={len(payload.get('images', []))} "
        f"annotations={len(payload.get('annotations', []))} "
        f"regions={len(payload.get('regions', []))} "
        f"warnings={counts['warning']} errors={counts['error']}"
    )
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
