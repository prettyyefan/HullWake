#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

FORBIDDEN_DIRECTORIES = {"data", "dataset", "datasets", "outputs", "checkpoints", "weights"}
FORBIDDEN_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".mp4",
    ".avi",
    ".mov",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".npy",
    ".npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a HullWake source-only release tree")
    parser.add_argument("root", nargs="?", default=".")
    return parser.parse_args()


def main() -> int:
    root = Path(parse_args().root).resolve()
    failures = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in FORBIDDEN_DIRECTORIES:
            failures.append(f"forbidden data/output directory: {relative}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"forbidden binary/data extension: {relative}")
        if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
            failures.append(f"unexpected file larger than 5 MiB: {relative}")
    if failures:
        print("\n".join(failures))
        return 1
    file_count = sum(path.is_file() for path in root.rglob("*"))
    print(f"release check passed: {file_count} files, no data/checkpoints/media")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
