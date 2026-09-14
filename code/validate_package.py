#!/usr/bin/env python3
"""Validate patient/image manifests and package file counts without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--decode-sample", type=int, default=30)
    args = parser.parse_args()
    patient_manifest = args.data_root / "patient_manifest.csv"
    image_manifest = args.data_root / "image_manifest.csv"
    patients = []
    with patient_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        patients = list(csv.DictReader(handle))
    images = []
    with image_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        images = list(csv.DictReader(handle))
    if len(patients) != 290:
        raise ValueError(f"Expected 290 patients, found {len(patients)}")
    if len(images) != 3706:
        raise ValueError(f"Expected 3706 images, found {len(images)}")
    patient_ids = [int(row["patient_id"]) for row in patients]
    if sorted(patient_ids) != list(range(1, 291)):
        raise ValueError("Patient IDs must be exactly 1..290")
    patient_lookup = {int(row["patient_id"]): row for row in patients}
    image_counts = defaultdict(int)
    missing = []
    for row in images:
        patient_id = int(row["patient_id"])
        if patient_id not in patient_lookup:
            raise ValueError(f"Unknown patient_id in image manifest: {patient_id}")
        if row["split"] != patient_lookup[patient_id]["split"] or row["label"] != patient_lookup[patient_id]["label"]:
            raise ValueError(f"Split/label mismatch for patient {patient_id}")
        image_counts[patient_id] += 1
        full_path = args.data_root / row["image_relpath"]
        if not full_path.is_file():
            missing.append(str(full_path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image files")
    for row in patients:
        patient_id = int(row["patient_id"])
        if image_counts[patient_id] != int(row["image_count"]):
            raise ValueError(f"Image count mismatch for patient {patient_id}")
    split_counts = Counter((row["split"], int(row["label"])) for row in patients)
    expected = {
        ("train", 0): 145, ("train", 1): 29,
        ("val", 0): 48, ("val", 1): 10,
        ("test", 0): 48, ("test", 1): 10,
    }
    if split_counts != Counter(expected):
        raise ValueError(f"Unexpected patient split counts: {split_counts}")
    sample = images[::max(1, len(images) // max(1, args.decode_sample))][:args.decode_sample]
    for row in sample:
        with Image.open(args.data_root / row["image_relpath"]) as image:
            image.verify()
    manifest_hash = hashlib.sha256(image_manifest.read_bytes()).hexdigest().upper()
    summary = {
        "status": "valid",
        "patients": len(patients),
        "images": len(images),
        "split_label_counts": {f"{key[0]}_label{key[1]}": value for key, value in sorted(split_counts.items())},
        "decoded_sample": len(sample),
        "image_manifest_sha256": manifest_hash,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
