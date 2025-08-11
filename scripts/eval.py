"""Evaluation script for trained FSNN models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from fsnn import FSNNMLP, GrowthConfig, PruneConfig, count_flops, get_dataloaders, measure_latency, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FSNN evaluation")
    parser.add_argument("--config", type=str, required=True, help="YAML config used for training")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(cfg, input_dim, num_classes, ckpt_path, device):
    model = FSNNMLP(
        input_dim,
        cfg["model"].get("hidden_features", 128),
        num_classes,
        GrowthConfig(**cfg["model"].get("growth", {})),
        PruneConfig(**cfg["model"].get("prune", {})),
        cfg["model"].get("complexity_lambda", 0.0),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model


def evaluate(model, loader, device):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb = xb.view(xb.size(0), -1)
            logits = model(xb)
            loss += F.cross_entropy(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(args.device)
    set_seed(args.seed)

    data_root = Path(cfg.get("data_root", "./data"))
    train_loader, val_loader, input_dim, num_classes = get_dataloaders(
        cfg["dataset"], data_root, batch_size=cfg.get("batch_size", 64)
    )
    model = load_model(cfg, input_dim, num_classes, args.checkpoint, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    dummy = torch.randn(1, input_dim, device=device)
    flops = count_flops(model, dummy)
    latency = measure_latency(lambda: model(dummy), repeats=20, warmup=5)
    print(
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} flops={flops} latency={latency*1000:.3f}ms"
    )


if __name__ == "__main__":
    main()
