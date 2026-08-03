from __future__ import annotations

import importlib
import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    failed = False
    for name, expected in (
        ("torch", "2.2.2"),
        ("torchvision", "0.17.2"),
        ("numpy", "1.26.4"),
        ("PIL", "10.2.0"),
        ("yaml", "6.0.1"),
        ("pycocotools", "2.0.7"),
    ):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "installed")
            status = "OK" if str(version).startswith(expected) else f"expected {expected}"
            print(f"{name}: {version} [{status}]")
        except Exception as exc:  # pragma: no cover - diagnostic CLI
            failed = True
            print(f"{name}: MISSING ({exc})")
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA runtime: {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    except Exception:
        pass
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
