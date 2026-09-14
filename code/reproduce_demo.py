"""Offline, deterministic CPU demonstration of the original ResNet-18 MIL model.

All images and labels are synthetic. Outputs are execution checks, not medical
performance estimates. No private data or pretrained downloads are required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_resnet_mil import (
    GatedAttentionMIL,
    choose_youden_threshold,
    metric_dict,
    set_seed,
)

SEED = 20260811


def synthetic_bags(seed: int, count: int = 8):
    """Produce two RGB noise images per bag, with a label-dependent patch."""
    generator = torch.Generator().manual_seed(seed)
    labels = [index % 2 for index in range(count)]
    bags = []
    for label in labels:
        images = torch.rand(2, 3, 64, 64, generator=generator) * 0.15
        channel = 0 if label == 0 else 2
        images[:, channel, 16:48, 16:48] += 0.75
        bags.append(images)
    return bags, labels


@torch.no_grad()
def probabilities(model, bags):
    model.eval()
    values = []
    attentions = []
    for bag in bags:
        logit, attention = model(bag, encode_chunk=2)
        values.append(float(logit.sigmoid()))
        attentions.append(attention.tolist())
    return values, attentions


def run_demo(output_dir: Path, seed: int = SEED, epochs: int = 6):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    set_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_bags, train_labels = synthetic_bags(seed + 1)
    val_bags, val_labels = synthetic_bags(seed + 2)
    test_bags, test_labels = synthetic_bags(seed + 3)
    model = GatedAttentionMIL(
        "resnet18", attention_dim=16, dropout=0.0, pretrained=False
    )
    # Freeze the random encoder to make the CPU demonstration fast.
    model.set_encoder_trainable(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.003,
    )
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()
        losses = []
        for bag, label in zip(train_bags, train_labels):
            optimizer.zero_grad(set_to_none=True)
            logit, _ = model(bag, encode_chunk=2)
            loss = criterion(logit, torch.tensor(float(label)))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_probabilities, _ = probabilities(model, val_bags)
        val_metrics = metric_dict(val_labels, val_probabilities, 0.5)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_ap": val_metrics["pr_auc"],
        })
    # Fixed epoch count; validation only selects the decision threshold.
    val_probabilities, _ = probabilities(model, val_bags)
    threshold = choose_youden_threshold(val_labels, val_probabilities)
    test_probabilities, attentions = probabilities(model, test_bags)
    assert all(np.isclose(sum(weights), 1.0) for weights in attentions)
    assert all(0 <= value <= 1 for value in test_probabilities)
    result = {
        "data_source": "synthetic_only_not_clinical_results",
        "seed": seed,
        "device": "cpu",
        "backbone": "resnet18",
        "pretrained": False,
        "encoder_frozen": True,
        "epochs": epochs,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "history": history,
        "validation": metric_dict(val_labels, val_probabilities, threshold),
        "test": metric_dict(test_labels, test_probabilities, threshold),
        "test_labels": test_labels,
        "test_probabilities": test_probabilities,
        "test_attention": attentions,
    }
    (output_dir / "demo_results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    result = run_demo(args.output_dir, seed=args.seed, epochs=args.epochs)
    print(json.dumps({"source": result["data_source"], "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
