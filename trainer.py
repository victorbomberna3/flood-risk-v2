"""
trainer.py — Training loop with mixed precision, spatial CV, and best-model checkpointing.
"""
from __future__ import annotations
from pathlib import Path
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, cohen_kappa_score


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    t = config["training"]
    if t["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=t["lr"],
            weight_decay=t["weight_decay"],
        )
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(f"Unknown optimizer: {t['optimizer']}")


def build_scheduler(opt, config: dict, steps_per_epoch: int):
    t = config["training"]
    total_steps = t["epochs"] * steps_per_epoch
    warmup_steps = t["warmup_epochs"] * steps_per_epoch
    if t["scheduler"] == "cosine":
        # Linear warmup → cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    if t["scheduler"] == "none":
        return None
    raise ValueError(f"Unknown scheduler: {t['scheduler']}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_tasks: int = 5,
    n_classes: int = 5,
) -> dict:
    """Compute QWK and Macro F1 per task on a dataset."""
    model.eval()
    all_preds = [[] for _ in range(n_tasks)]
    all_trues = [[] for _ in range(n_tasks)]

    for x, t, vm in loader:
        x = x.to(device, non_blocking=True)
        outputs = model(x)
        vm_np = vm.numpy()
        for i, logits in enumerate(outputs):
            preds = logits.argmax(dim=1).cpu().numpy()
            trues = t[:, i].numpy()
            mask = (trues >= 0) & vm_np
            all_preds[i].append(preds[mask])
            all_trues[i].append(trues[mask])

    results = {}
    per_task_f1 = []
    per_task_qwk = []
    labels = list(range(n_classes))
    for i in range(n_tasks):
        if len(all_preds[i]) == 0:
            results[f"task_{i}_f1"] = 0.0
            results[f"task_{i}_qwk"] = 0.0
            per_task_f1.append(0.0)
            per_task_qwk.append(0.0)
            continue
        p = np.concatenate(all_preds[i])
        t_ = np.concatenate(all_trues[i])
        if len(t_) == 0:
            f1, qwk = 0.0, 0.0
        else:
            f1 = float(f1_score(t_, p, labels=labels, average="macro", zero_division=0))
            try:
                qwk = float(cohen_kappa_score(t_, p, weights="quadratic", labels=labels))
            except Exception:
                qwk = 0.0
        results[f"task_{i}_f1"] = f1
        results[f"task_{i}_qwk"] = qwk
        per_task_f1.append(f1)
        per_task_qwk.append(qwk)

    results["macro_f1_mean"] = float(np.mean(per_task_f1))
    results["qwk_mean"] = float(np.mean(per_task_qwk))
    return results


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    config: dict,
    device: torch.device,
    checkpoint_dir: str | Path,
    log_callback=None,
) -> dict:
    """
    Train U-Net end-to-end. Returns dict with best metrics + checkpoint path.
    """
    t = config["training"]
    epochs = t["epochs"]
    grad_clip = t.get("grad_clip", 0.0)
    use_amp = t.get("mixed_precision", True)

    opt = build_optimizer(model, config)
    sched = build_scheduler(opt, config, steps_per_epoch=len(train_loader))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "model_best.pt"
    best_metric = -1.0

    history = []

    print(f"\n=== Training: {epochs} epochs, {len(train_loader)} batches/epoch ===")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0

        for batch_idx, (x, tgt, vm) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            vm = vm.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = model(x)
                loss, per_task = loss_fn(outputs, tgt, vm)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(1, n_batches)
        val_metrics = evaluate(model, val_loader, device, n_tasks=len(config["targets"]), n_classes=config["n_classes"])

        elapsed = time.time() - t0
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": elapsed,
            **val_metrics,
        }
        history.append(log)
        n_t = len(config["targets"])
        print(
            f"  Ep {epoch:2d}/{epochs}  "
            f"loss={train_loss:.4f}  "
            f"val_QWK={val_metrics['qwk_mean']:.4f}  "
            f"val_F1={val_metrics['macro_f1_mean']:.4f}  "
            f"per-task_QWK={[f\"{val_metrics[f'task_{i}_qwk']:.3f}\" for i in range(n_t)]}  "
            f"[{elapsed:.0f}s]"
        )

        if log_callback is not None:
            log_callback(log)

        # Save best checkpoint based on QWK (ordinal-aware metric)
        if val_metrics["qwk_mean"] > best_metric:
            best_metric = val_metrics["qwk_mean"]
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
            }, best_path)
            print(f"    NEW BEST  → {best_path}")

    # Save history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_metric": best_metric,
        "best_checkpoint": str(best_path),
        "final_metrics": history[-1],
        "history": history,
    }
