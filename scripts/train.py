"""Training script for FSNN models using YAML configs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

from fsnn import FSNNMLP, GrowthConfig, PruneConfig, count_flops, get_dataloaders, set_seed


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSNN training entry point")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ---------------------------------------------------------------------------


def build_model(cfg: Dict[str, Any], input_dim: int, num_classes: int) -> FSNNMLP:
    growth = GrowthConfig(**cfg.get("growth", {}))
    prune = PruneConfig(**cfg.get("prune", {}))
    hidden = cfg.get("hidden_features", 128)
    complexity_lambda = cfg.get("complexity_lambda", 0.0)
    return FSNNMLP(
        in_features=input_dim,
        hidden_features=hidden,
        out_features=num_classes,
        growth=growth,
        prune=prune,
        complexity_lambda=complexity_lambda,
    )


# ---------------------------------------------------------------------------


def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xb = xb.view(xb.size(0), -1)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb) + model.complexity_penalty()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.view(xb.size(0), -1)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    set_seed(args.seed)

    data_root = Path(cfg.get("data_root", "./data"))
    train_loader, val_loader, input_dim, num_classes = get_dataloaders(
        cfg["dataset"], data_root, batch_size=cfg.get("batch_size", 64)
    )

    model = build_model(cfg["model"], input_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    out_dir = Path(cfg.get("output", "runs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="") as fcsv:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_acc",
            "hidden_size",
            "flops",
            "grown",
            "pruned",
        ])
        epochs = cfg.get("epochs", 1)
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, opt, device)
            val_loss, val_acc = evaluate(model, val_loader, device)
            grown, pruned = model.update_controller(val_loss)
            dummy = torch.randn(1, input_dim, device=device)
            flops = count_flops(model, dummy)
            csv_writer.writerow(
                [epoch, train_loss, val_loss, val_acc, model.fc1.out_features, flops, grown, pruned]
            )
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            writer.add_scalar("model/hidden", model.fc1.out_features, epoch)
            writer.add_scalar("model/flops", flops, epoch)
    torch.save(model.state_dict(), out_dir/"model.pt")
    writer.close()


if __name__ == "__main__":
    main()
