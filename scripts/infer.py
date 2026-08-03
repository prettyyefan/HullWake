#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from hullwake.config import load_config
from hullwake.modeling import build_model
from hullwake.utils.checkpoint import load_model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HullWake on one local image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-json")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    device = torch.device(config["train"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    with Image.open(args.image) as source:
        display = source.convert("RGB")
    array = np.asarray(display, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(device)
    model = build_model(config).to(device).eval()
    load_model_checkpoint(args.checkpoint, model, map_location=device)
    with torch.inference_mode():
        output = model([tensor])[0]
    keep = output["scores"] >= float(args.score_threshold)
    draw = ImageDraw.Draw(display)
    records = []
    for box, label, score, attenuated in zip(
        output["boxes"][keep].cpu(),
        output["labels"][keep].cpu(),
        output["scores"][keep].cpu(),
        output["attenuated_scores"][keep].cpu(),
    ):
        coordinates = [float(value) for value in box.tolist()]
        draw.rectangle(coordinates, outline=(0, 220, 90), width=3)
        text = f"vessel {float(score):.3f} / att {float(attenuated):.3f}"
        draw.text((coordinates[0], max(0.0, coordinates[1] - 14)), text, fill=(0, 220, 90))
        records.append(
            {
                "box_xyxy": coordinates,
                "label": int(label),
                "score": float(score),
                "attenuated_score": float(attenuated),
                "wake_drop": float(score - attenuated),
            }
        )
    destination = Path(args.output_image)
    destination.parent.mkdir(parents=True, exist_ok=True)
    display.save(destination)
    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2)
            handle.write("\n")
    print(f"Saved {len(records)} detections to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
