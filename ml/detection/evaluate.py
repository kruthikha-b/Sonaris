"""
Sonaris P1: AI/ML Detection Module
Model Evaluation Pipeline for Test / Validation Splits
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(
    model_path="models/weights/best.pt",
    data_yaml="/tmp/sonaris_p1_dataset/data.yaml",
    split="test",
    output_json=None,
):
    model_file = Path(model_path).resolve()
    data_file = Path(data_yaml).resolve()

    if not model_file.exists():
        raise FileNotFoundError(f"Model weights not found at: {model_file}")

    if not data_file.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_file}")

    print("=" * 60)
    print(f"SONARIS P1: MODEL EVALUATION ({split.upper()} SPLIT)")
    print("=" * 60)
    print(f"Model weights : {model_file}")
    print(f"Dataset config: {data_file}")
    print(f"Split target  : {split}")
    print("=" * 60)

    model = YOLO(str(model_file))

    metrics = model.val(
        data=str(data_file),
        split=split,
        verbose=True,
    )

    overall_precision = float(metrics.box.mp)
    overall_recall = float(metrics.box.mr)
    overall_map50 = float(metrics.box.map50)
    overall_map50_95 = float(metrics.box.map)

    speed_info = {}

    if hasattr(metrics, "speed") and metrics.speed:
        speed_info = {
            "preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
            "inference_ms": float(metrics.speed.get("inference", 0.0)),
            "postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
        }

        total_latency_ms = (
            speed_info["preprocess_ms"]
            + speed_info["inference_ms"]
            + speed_info["postprocess_ms"]
        )

        speed_info["total_latency_ms"] = round(total_latency_ms, 2)
        speed_info["fps"] = (
            round(1000.0 / total_latency_ms, 1)
            if total_latency_ms > 0
            else 0.0
        )

    class_names = metrics.names

    per_class_metrics = {}

    precision = metrics.box.p
    recall = metrics.box.r
    ap50 = metrics.box.ap50
    ap = metrics.box.ap
    class_indices = metrics.box.ap_class_index

    for idx, cls_idx in enumerate(class_indices):
        cls_idx = int(cls_idx)
        cls_name = class_names.get(cls_idx, f"class_{cls_idx}")

        per_class_metrics[cls_name] = {
            "class_id": cls_idx,
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "map50": round(float(ap50[idx]), 4),
            "map50_95": round(float(ap[idx]), 4),
        }

    results = {
        "split": split,
        "model_path": str(model_file),
        "data_path": str(data_file),
        "precision": round(overall_precision, 4),
        "recall": round(overall_recall, 4),
        "map50": round(overall_map50, 4),
        "map50_95": round(overall_map50_95, 4),
        "speed": speed_info,
        "per_class": per_class_metrics,
    }

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall mAP@0.50  : {overall_map50:.4f}")
    print(f"Overall mAP@50-95 : {overall_map50_95:.4f}")

    if speed_info:
        print(
            f"Inference Latency : "
            f"{speed_info['inference_ms']:.2f} ms/img "
            f"(FPS: {speed_info['fps']})"
        )

    print("\n--- Per-Class Performance ---")

    header = (
        f"{'Class Name':<22} | "
        f"{'Class ID':<8} | "
        f"{'Precision':<10} | "
        f"{'Recall':<10} | "
        f"{'mAP50':<10} | "
        f"{'mAP50-95':<10}"
    )

    print(header)
    print("-" * len(header))

    for cname, cdata in sorted(
        per_class_metrics.items(),
        key=lambda x: x[1]["class_id"],
    ):
        print(
            f"{cname:<22} | "
            f"{cdata['class_id']:<8} | "
            f"{cdata['precision']:<10.4f} | "
            f"{cdata['recall']:<10.4f} | "
            f"{cdata['map50']:<10.4f} | "
            f"{cdata['map50_95']:<10.4f}"
        )

    print("=" * 60)

    if output_json:
        out_path = Path(output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)

        print(f"Detailed evaluation metrics saved to: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on Sonaris dataset split"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/weights/best.pt",
        help="Path to model weights",
    )

    parser.add_argument(
        "--data",
        type=str,
        default="/tmp/sonaris_p1_dataset/data.yaml",
        help="Path to data.yaml",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Split to evaluate",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="reports/baseline_test_evaluation.json",
        help="Path to save output JSON",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
