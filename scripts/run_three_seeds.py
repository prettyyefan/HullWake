#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from hullwake.config import load_config
from hullwake.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and summarize independent HullWake seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and value is not None
        }
    )
    summary: dict[str, Any] = {"number_of_completed_seeds": len(rows), "metrics": {}}
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if len(values) != len(rows):
            continue
        summary["metrics"][key] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def main() -> int:
    args = parse_args()
    base = load_config(args.config, args.overrides)
    root = Path(base["experiment"]["output_root"]) / base["experiment"]["name"]
    train_script = Path(__file__).with_name("train.py")
    rows = []
    for seed in args.seeds:
        command = [
            sys.executable,
            str(train_script),
            "--config",
            args.config,
            "--set",
            f"experiment.seed={seed}",
        ]
        for override in args.overrides:
            command.extend(("--set", override))
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
            metrics_path = root / f"seed_{seed}" / "metrics_val_latest.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Completed seed has no metrics: {metrics_path}")
            with metrics_path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
            row["seed"] = seed
            rows.append(row)
    if not args.dry_run:
        summary = summarize(rows)
        summary["seeds"] = args.seeds
        write_json(summary, root / "three_seed_summary.json")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
