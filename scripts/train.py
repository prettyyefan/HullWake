#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hullwake.config import load_config
from hullwake.data import build_dataset, collate_fn
from hullwake.engine import train
from hullwake.evaluation import CocoWakeEvaluator
from hullwake.modeling import build_model
from hullwake.utils.seed import seed_everything, worker_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HullWake")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed, bool(config["experiment"]["deterministic"]))
    device = torch.device(config["train"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --set train.device=cpu for a test")

    train_dataset = build_dataset(config, "train", training=True)
    val_dataset = build_dataset(config, "val", training=False)
    expected_classes = len(train_dataset.category_id_to_label) + 1
    if expected_classes != int(config["data"]["num_classes"]):
        raise ValueError(
            f"data.num_classes={config['data']['num_classes']} but annotations require {expected_classes}"
        )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["train"]["images_per_gpu"]),
        shuffle=True,
        num_workers=int(config["data"]["workers"]),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"]["workers"]) > 0,
        worker_init_fn=worker_seed,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"]["workers"]),
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config["data"]["workers"]) > 0,
        worker_init_fn=worker_seed,
    )
    model = build_model(config)
    evaluator = CocoWakeEvaluator(
        val_dataset.coco_data,
        val_dataset.label_to_category_id,
        config["evaluation"],
    )
    output = (
        Path(config["experiment"]["output_root"])
        / config["experiment"]["name"]
        / f"seed_{seed}"
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        evaluator=evaluator,
        device=device,
        config=config,
        output_directory=output,
        resume=args.resume,
    )
    print(f"Completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
