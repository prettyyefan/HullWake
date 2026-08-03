from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from hullwake.evaluation import CocoWakeEvaluator
from hullwake.utils.checkpoint import load_training_checkpoint, save_checkpoint
from hullwake.utils.io import JsonlLogger, runtime_manifest, write_json


def move_targets(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    warmup_iterations: int,
    warmup_factor: float,
    milestones: list[int],
    gamma: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    milestone_steps = [int(epoch) * steps_per_epoch for epoch in milestones]

    def multiplier(step: int) -> float:
        if warmup_iterations > 0 and step < warmup_iterations:
            alpha = float(step) / float(max(1, warmup_iterations))
            warmup = float(warmup_factor) * (1.0 - alpha) + alpha
        else:
            warmup = 1.0
        decays = sum(step >= milestone for milestone in milestone_steps)
        return warmup * float(gamma) ** decays

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    evaluator: CocoWakeEvaluator,
    device: torch.device,
) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    model.eval()
    predictions: dict[int, dict[str, torch.Tensor]] = {}
    for images, targets in tqdm(loader, desc="evaluate", leave=False):
        outputs = model([image.to(device, non_blocking=True) for image in images])
        for output, target in zip(outputs, targets):
            image_id = int(target["image_id"].item())
            predictions[image_id] = {
                key: value.detach().cpu() for key, value in output.items() if torch.is_tensor(value)
            }
    metrics = evaluator.evaluate(predictions)
    return metrics, predictions


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: dict[str, Any],
    logger: JsonlLogger,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = int(config["train"]["gradient_accumulation_steps"])
    log_every = int(config["train"]["log_every"])
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    clip_norm = float(config["train"]["gradient_clip_norm"])
    running: dict[str, float] = {}
    batches = len(loader)
    started = time.perf_counter()

    for batch_index, (images, targets) in enumerate(
        tqdm(loader, desc=f"train {epoch + 1}", leave=False)
    ):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets(targets, device)
        group_start = (batch_index // accumulation) * accumulation
        group_size = min(accumulation, batches - group_start)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())
            scaled_loss = total_loss / float(group_size)
        if not torch.isfinite(total_loss):
            values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}: {values}")
        scaler.scale(scaled_loss).backward()
        should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == batches
        if should_step:
            scaler.unscale_(optimizer)
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_values = {key: float(value.detach().cpu()) for key, value in loss_dict.items()}
        batch_values["loss_total"] = float(total_loss.detach().cpu())
        for key, value in batch_values.items():
            running[key] = running.get(key, 0.0) + value
        if (batch_index + 1) % log_every == 0 or batch_index + 1 == batches:
            logger.log(
                {
                    "event": "train_batch",
                    "epoch": epoch,
                    "batch": batch_index,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **batch_values,
                }
            )
    summary = {key: value / max(1, batches) for key, value in running.items()}
    summary["seconds"] = time.perf_counter() - started
    summary["learning_rate"] = float(optimizer.param_groups[0]["lr"])
    return summary


def train(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    evaluator: CocoWakeEvaluator,
    device: torch.device,
    config: dict[str, Any],
    output_directory: str | Path,
    resume: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_json(runtime_manifest(config), output / "run_manifest.json")
    logger = JsonlLogger(output / "metrics.jsonl")
    train_config = config["train"]
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_config["learning_rate"]),
        momentum=float(train_config["momentum"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    accumulation = int(train_config["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = int(math.ceil(len(train_loader) / accumulation))
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=optimizer_steps_per_epoch,
        warmup_iterations=int(train_config["warmup_iterations"]),
        warmup_factor=float(train_config["warmup_factor"]),
        milestones=list(train_config["lr_milestones"]),
        gamma=float(train_config["lr_gamma"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(train_config["amp"]) and device.type == "cuda"
    )
    model.to(device)
    start_epoch = 0
    best_metric = -math.inf
    if resume is not None:
        start_epoch, best_metric, checkpoint_config = load_training_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        if checkpoint_config != config:
            logger.log({"event": "resume_config_difference", "checkpoint": str(resume)})

    latest_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
            logger=logger,
        )
        logger.log({"event": "train_epoch", "epoch": epoch, **train_metrics})
        should_evaluate = (epoch + 1) % int(train_config["evaluate_every"]) == 0
        if should_evaluate:
            latest_metrics, _ = evaluate_model(model, val_loader, evaluator, device)
            write_json(latest_metrics, output / f"metrics_val_epoch_{epoch + 1:04d}.json")
            write_json(latest_metrics, output / "metrics_val_latest.json")
            logger.log({"event": "validation", "epoch": epoch, **latest_metrics})
            selection_metric = latest_metrics.get("WG_AP")
            if selection_metric is None:
                selection_metric = latest_metrics.get("AP")
            if selection_metric is not None and float(selection_metric) > best_metric:
                best_metric = float(selection_metric)
                save_checkpoint(
                    output / "checkpoint_best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    best_metric=best_metric,
                    config=config,
                )
        if (epoch + 1) % int(train_config["save_every"]) == 0:
            save_checkpoint(
                output / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
            )
    summary = {"best_selection_metric": best_metric, "latest_validation": latest_metrics}
    write_json(summary, output / "training_summary.json")
    return summary
