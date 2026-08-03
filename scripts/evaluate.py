#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hullwake.config import load_config
from hullwake.data import build_dataset, collate_fn
from hullwake.engine import evaluate_model
from hullwake.evaluation import CocoWakeEvaluator, predictions_to_coco
from hullwake.modeling import build_model
from hullwake.utils.checkpoint import load_model_checkpoint
from hullwake.utils.io import write_json
from hullwake.utils.seed import seed_everything, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a HullWake checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output")
    parser.add_argument("--save-predictions")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed_everything(
        int(config["experiment"]["seed"]), bool(config["experiment"]["deterministic"])
    )
    device = torch.device(config["train"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset = build_dataset(config, args.split, training=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"]["workers"]),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"]["workers"]) > 0,
        worker_init_fn=worker_seed,
    )
    model = build_model(config).to(device)
    load_model_checkpoint(args.checkpoint, model, map_location=device)
    evaluator = CocoWakeEvaluator(
        dataset.coco_data, dataset.label_to_category_id, config["evaluation"]
    )
    metrics, predictions = evaluate_model(model, loader, evaluator, device)
    output_path = args.output or str(Path(args.checkpoint).with_name(f"metrics_{args.split}.json"))
    write_json(metrics, output_path)
    if args.save_predictions:
        write_json(
            predictions_to_coco(predictions, dataset.label_to_category_id),
            args.save_predictions,
        )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
