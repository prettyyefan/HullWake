from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigError(ValueError):
    """Raised when a HullWake configuration is incomplete or inconsistent."""


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_recursive(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"Cyclic _base_ configuration: {chain}")
    if not path.is_file():
        raise ConfigError(f"Configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {path}")
    base_ref = payload.get("_base_")
    if base_ref is None:
        return payload
    base_refs = [base_ref] if isinstance(base_ref, str) else base_ref
    if not isinstance(base_refs, list) or not all(isinstance(x, str) for x in base_refs):
        raise ConfigError("_base_ must be a path string or a list of path strings")
    merged: dict[str, Any] = {}
    for ref in base_refs:
        base_payload = _load_recursive(path.parent / ref, (*stack, path))
        merged = _deep_merge(merged, base_payload)
    return _deep_merge(merged, payload)


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = config
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    for expression in overrides:
        if "=" not in expression:
            raise ConfigError(f"Override must be KEY=VALUE, got: {expression}")
        key, raw_value = expression.split("=", 1)
        if not key.strip():
            raise ConfigError(f"Override has an empty key: {expression}")
        _set_dotted(result, key.strip(), yaml.safe_load(raw_value))
    return result


def validate_config(config: dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "model", "loss", "train", "evaluation")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    model = config["model"]
    loss = config["loss"]
    train = config["train"]
    data = config["data"]

    points = int(model["corridor_points_per_direction"])
    grid = int(model["corridor_grid_size"])
    if points != grid * grid:
        raise ConfigError(
            "corridor_points_per_direction must equal corridor_grid_size squared "
            f"(got {points} and {grid})"
        )
    if len(model["fpn_levels"]) != len(model["fpn_strides"]):
        raise ConfigError("model.fpn_levels and model.fpn_strides must have equal lengths")
    if int(data["num_classes"]) < 2:
        raise ConfigError("data.num_classes includes background and must be at least 2")
    if not 0.0 <= float(loss["lambda_att"]) <= 1.0:
        raise ConfigError("loss.lambda_att must be in [0, 1]")
    if float(loss["epsilon"]) <= 0:
        raise ConfigError("loss.epsilon must be positive")
    actual = int(train["images_per_gpu"]) * int(train["gradient_accumulation_steps"])
    if actual != int(train["effective_batch_size"]):
        raise ConfigError(
            "train.effective_batch_size must equal images_per_gpu * "
            f"gradient_accumulation_steps (expected {actual})"
        )
    if sorted(train["lr_milestones"]) != list(train["lr_milestones"]):
        raise ConfigError("train.lr_milestones must be sorted")
    if train["lr_milestones"] and max(train["lr_milestones"]) >= int(train["epochs"]):
        raise ConfigError("Every LR milestone must be smaller than train.epochs")


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config = _load_recursive(Path(path))
    config = apply_overrides(config, overrides)
    validate_config(config)
    config.setdefault("runtime", {})["config_file"] = str(Path(path).resolve())
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
