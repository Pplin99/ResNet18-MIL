#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--sgd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, directory in (("adamw", args.adamw), ("sgd", args.sgd)):
        summary = json.loads((directory / "summary_metrics.json").read_text(encoding="utf-8"))
        rows.append({
            "optimizer": name,
            "best_epoch": summary["best_epoch"],
            "val_pr_auc": summary["validation"]["pr_auc"],
            "val_roc_auc": summary["validation"]["roc_auc"],
            "test_pr_auc": summary["test"]["pr_auc"],
            "test_roc_auc": summary["test"]["roc_auc"],
            "test_sensitivity": summary["test"]["sensitivity"],
            "test_specificity": summary["test"]["specificity"],
            "test_brier": summary["test"]["brier"],
        })
    winner = max(rows, key=lambda row: row["val_pr_auc"])["optimizer"]
    result = {"selection_rule": "highest validation PR-AUC", "selected_optimizer": winner, "models": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
