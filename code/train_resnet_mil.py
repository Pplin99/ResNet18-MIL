#!/usr/bin/env python3
"""Patient-level multi-image ResNet-50 with attention MIL pooling."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), required=True)
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--max-train-images", type=int, default=16)
    parser.add_argument("--max-eval-images", type=int, default=0,
                        help="0 means use every image for validation/test")
    parser.add_argument("--encode-chunk", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--attention-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--sgd-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Letterbox:
    def __init__(self, size: int) -> None:
        self.size = (size, size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image.convert("RGB"), self.size, method=Image.Resampling.BILINEAR,
            color=(0, 0, 0), centering=(0.5, 0.5)
        )


def build_transform(image_size: int, train: bool):
    steps = [Letterbox(image_size)]
    if train:
        steps.extend([
            transforms.RandomRotation(7, interpolation=transforms.InterpolationMode.BILINEAR, fill=0),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ])
    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transforms.Compose(steps)


class PatientBagDataset(Dataset):
    def __init__(self, data_root: Path, split: str, image_size: int,
                 max_images: int, seed: int) -> None:
        self.data_root = data_root
        self.split = split
        self.max_images = max_images
        self.seed = seed
        self.epoch = 0
        self.transform = build_transform(image_size, train=(split == "train"))
        manifest_path = data_root / "image_manifest.csv"
        grouped: dict[int, dict] = defaultdict(lambda: {"label": None, "paths": []})
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] != split:
                    continue
                patient_id = int(row["patient_id"])
                label = int(row["label"])
                record = grouped[patient_id]
                if record["label"] is None:
                    record["label"] = label
                if record["label"] != label:
                    raise ValueError(f"Conflicting labels for patient {patient_id}")
                record["paths"].append(row["image_relpath"])
        self.records = []
        for patient_id in sorted(grouped):
            paths = sorted(grouped[patient_id]["paths"])
            if not paths:
                raise ValueError(f"Patient {patient_id} has no images")
            self.records.append((patient_id, int(grouped[patient_id]["label"]), paths))
        if not self.records:
            raise ValueError(f"No records for split={split}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _select_paths(self, patient_id: int, paths: list[str]) -> list[str]:
        if self.max_images <= 0 or len(paths) <= self.max_images:
            return paths
        if self.split == "train":
            rng = random.Random(self.seed + self.epoch * 1_000_003 + patient_id)
            indices = sorted(rng.sample(range(len(paths)), self.max_images))
        else:
            indices = np.linspace(0, len(paths) - 1, self.max_images).round().astype(int).tolist()
        return [paths[index] for index in indices]

    def __getitem__(self, index: int):
        patient_id, label, all_paths = self.records[index]
        selected_paths = self._select_paths(patient_id, all_paths)
        tensors = []
        for relative_path in selected_paths:
            image_path = self.data_root / relative_path
            with Image.open(image_path) as image:
                tensors.append(self.transform(image))
        return torch.stack(tensors), torch.tensor(float(label)), patient_id, selected_paths


def patient_collate(batch):
    if len(batch) != 1:
        raise ValueError("This implementation expects DataLoader batch_size=1 patient")
    return batch[0]


class GatedAttentionMIL(nn.Module):
    def __init__(self, backbone: str, attention_dim: int, dropout: float, pretrained: bool) -> None:
        super().__init__()
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.encoder = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.attention_v = nn.Linear(feature_dim, attention_dim)
        self.attention_u = nn.Linear(feature_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(feature_dim, 1))

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = trainable

    def forward(self, images: torch.Tensor, encode_chunk: int):
        features = []
        for start in range(0, images.shape[0], encode_chunk):
            features.append(self.encoder(images[start:start + encode_chunk]))
        features = torch.cat(features, dim=0)
        gated = torch.tanh(self.attention_v(features)) * torch.sigmoid(self.attention_u(features))
        scores = self.attention_w(gated).squeeze(1)
        attention = torch.softmax(scores, dim=0)
        pooled = torch.sum(attention.unsqueeze(1) * features, dim=0)
        logit = self.classifier(pooled).squeeze(0)
        return logit, attention


def choose_youden_threshold(labels: list[int], probabilities: list[float]) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    index = np.argmax((tpr - fpr)[finite])
    return float(thresholds[finite][index])


def metric_dict(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    predicted = [int(probability >= threshold) for probability in probabilities]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(labels),
        "positive": int(sum(labels)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "sensitivity": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else math.nan,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


@torch.no_grad()
def evaluate(model, loader, device, encode_chunk: int, criterion=None, attention_output: Path | None = None):
    model.eval()
    labels, probabilities, patient_ids = [], [], []
    total_loss = 0.0
    attention_rows = []
    for images, label, patient_id, relative_paths in loader:
        images = images.to(device, non_blocking=True)
        label = label.to(device)
        logit, attention = model(images, encode_chunk)
        if criterion is not None:
            total_loss += float(criterion(logit.view(1), label.view(1)).item())
        probability = float(torch.sigmoid(logit).item())
        labels.append(int(label.item()))
        probabilities.append(probability)
        patient_ids.append(int(patient_id))
        if attention_output is not None:
            for relative_path, weight in zip(relative_paths, attention.detach().cpu().tolist()):
                attention_rows.append((patient_id, relative_path, weight))
    if attention_output is not None:
        with attention_output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["patient_id", "image_relpath", "attention_weight"])
            writer.writerows(attention_rows)
    return {
        "loss": total_loss / len(loader),
        "labels": labels,
        "probabilities": probabilities,
        "patient_ids": patient_ids,
    }


def write_predictions(path: Path, result: dict, threshold: float) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "label", "probability", "prediction", "threshold"])
        for patient_id, label, probability in zip(
            result["patient_ids"], result["labels"], result["probabilities"]
        ):
            writer.writerow([patient_id, label, f"{probability:.10f}", int(probability >= threshold), f"{threshold:.10f}"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this training job")

    train_dataset = PatientBagDataset(args.data_root, "train", args.image_size, args.max_train_images, args.seed)
    val_dataset = PatientBagDataset(args.data_root, "val", args.image_size, args.max_eval_images, args.seed)
    test_dataset = PatientBagDataset(args.data_root, "test", args.image_size, args.max_eval_images, args.seed)
    expected = {"train": (174, 29), "val": (58, 10), "test": (58, 10)}
    for name, dataset in (("train", train_dataset), ("val", val_dataset), ("test", test_dataset)):
        positives = sum(record[1] for record in dataset.records)
        if (len(dataset), positives) != expected[name]:
            raise ValueError(f"Unexpected {name} split: patients={len(dataset)} positives={positives}")

    loader_kwargs = dict(batch_size=1, num_workers=args.num_workers, collate_fn=patient_collate,
                         pin_memory=True, persistent_workers=False)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = GatedAttentionMIL(
        args.backbone, args.attention_dim, args.dropout,
        pretrained=(args.weights == "imagenet")
    )
    model.to(device)
    model.set_encoder_trainable(args.freeze_epochs <= 0)
    train_positive = expected["train"][1]
    train_negative = expected["train"][0] - train_positive
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_negative / train_positive], device=device))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.sgd_lr, momentum=0.9,
                                    nesterov=True, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    config = vars(args).copy()
    config.update({
        "data_root": str(args.data_root), "output_dir": str(args.output_dir),
        "device": str(device), "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "train_pos_weight": train_negative / train_positive,
    })
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    history = []
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            model.set_encoder_trainable(True)
        train_dataset.set_epoch(epoch)
        model.train()
        if epoch <= args.freeze_epochs:
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (images, label, _patient_id, _relative_paths) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            label = label.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                logit, _attention = model(images, args.encode_chunk)
                loss = criterion(logit.view(1), label.view(1))
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
        val_result = evaluate(model, val_loader, device, args.encode_chunk, criterion)
        val_metrics = metric_dict(val_result["labels"], val_result["probabilities"], 0.5)
        score = val_metrics["pr_auc"]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "val_loss": val_result["loss"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_brier": val_metrics["brier"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "encoder_trainable": epoch > args.freeze_epochs,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        # Treat an absolute validation PR-AUC gain below 0.001 as no meaningful improvement.
        if score > best_score + 0.001:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_pr_auc": score, "config": config},
                       args.output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            break

    with (args.output_dir / "history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_result = evaluate(model, val_loader, device, args.encode_chunk, criterion,
                          args.output_dir / "val_attention.csv")
    threshold = choose_youden_threshold(val_result["labels"], val_result["probabilities"])
    test_result = evaluate(model, test_loader, device, args.encode_chunk, criterion,
                           args.output_dir / "test_attention.csv")
    val_metrics = metric_dict(val_result["labels"], val_result["probabilities"], threshold)
    test_metrics = metric_dict(test_result["labels"], test_result["probabilities"], threshold)
    write_predictions(args.output_dir / "val_predictions.csv", val_result, threshold)
    write_predictions(args.output_dir / "test_predictions.csv", test_result, threshold)
    summary = {
        "optimizer": args.optimizer,
        "best_epoch": best_epoch,
        "selection_metric": "validation_pr_auc",
        "best_validation_pr_auc": best_score,
        "validation": val_metrics,
        "test": test_metrics,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
