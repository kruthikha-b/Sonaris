"""
Sonaris P1: AI/ML Detection Module
Model Evaluation Pipeline for Test / Validation Splits
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def evaluate(model_path="models/weights/best.pt", data_yaml="/tmp/sonaris_p1_dataset/data.yaml", split="test", output_json=None):
    """
    Evaluate a trained YOLO model on the specified split (default: test).

    Args:
        model_path (str): Path to trained model weights (.pt).
        data_yaml (str): Path to dataset YAML configuration.
        split (str): Split to evaluate on ('test', 'val').
        output_json (str, optional): Destination path to save structured metrics.

    Returns:
        dict: Detailed evaluation metrics including overall and per-class performance.
    """
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
        verbose=True
    )

    # Extract overall metrics
    overall_precision = float(metrics.box.mp)
    overall_recall = float(metrics.box.mr)
    overall_map50 = float(metrics.box.map50)
    overall_map50_95 = float(metrics.box.map)

    # Extract speed / latency if available
    speed_info = {}
    if hasattr(metrics, "speed") and metrics.speed:
        speed_info = {
            "preprocess_ms": float(metrics.speed.get("preprocess", 0.0)),
            "inference_ms": float(metrics.speed.get("inference", 0.0)),
            "loss_ms": float(metrics.speed.get("loss", 0.0)),
            "postprocess_ms": float(metrics.speed.get("postprocess", 0.0)),
        }
        total_latency_ms = sum(speed_info.values())
        speed_info["total_latency_ms"] = round(total_latency_ms, 2)
        speed_info["fps"] = round(1000.0 / total_latency_ms, 1) if total_latency_ms > 0 else 0.0

    # Extract per-class metrics
    class_names = metrics.names if hasattr(metrics, "names") else {}
    per_class_metrics = {}

    if hasattr(metrics.box, "ap50") and hasattr(metrics.box, "ap"):
        # metrics.box.ap is an array of shape (num_classes,) containing mAP50-95
        # metrics.box.ap50 is an array containing mAP50 per class
        # metrics.box.p and metrics.box.r contain precision and recall per class
        classes_present = getattr(metrics.box, "classes", list(range(len(metrics.box.ap))))

        for idx, cls_idx in enumerate(classes_present):
            cls_name = class_names.get(cls_idx, f"class_{cls_idx}")
            p_val = float(metrics.box.p[idx]) if hasattr(metrics.box, "p") and len(metrics.box.p) > idx else 0.0
            r_val = float(metrics.box.r[idx]) if hasattr(metrics.box, "r") and len(metrics.box.r) > idx else 0.0
            ap50_val = float(metrics.box.ap50[idx]) if len(metrics.box.ap50) > idx else 0.0
            ap_val = float(metrics.box.ap[idx]) if len(metrics.box.ap) > idx else 0.0

            per_class_metrics[cls_name] = {
                "class_id": int(cls_idx),
                "precision": round(p_val, 4),
                "recall": round(r_val, 4),
                "map50": round(ap50_val, 4),
                "map50_95": round(ap_val, 4),
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

    # Print clean results table
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Overall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall mAP@0.50  : {overall_map50:.4f}")
    print(f"Overall mAP@50-95 : {overall_map50_95:.4f}")
    if speed_info:
        print(f"Inference Latency : {speed_info.get('inference_ms', 0.0):.2f} ms/img (FPS: {speed_info.get('fps', 0.0)})")

    print("\n--- Per-Class Performance ---")
    header = f"{'Class Name':<22} | {'Class ID':<8} | {'Precision':<10} | {'Recall':<10} | {'mAP50':<10} | {'mAP50-95':<10}"
    print(header)
    print("-" * len(header))
    for cname, cdata in sorted(per_class_metrics.items(), key=lambda x: x[1]["class_id"]):
        print(f"{cname:<22} | {cdata['class_id']:<8} | {cdata['precision']:<10.4f} | {cdata['recall']:<10.4f} | {cdata['map50']:<10.4f} | {cdata['map50_95']:<10.4f}")
    print("=" * 60)

    if output_json:
        out_p = Path(output_json).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed evaluation metrics saved to: {out_p}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on Sonaris dataset split")
    parser.add_argument("--model", type=str, default="models/weights/best.pt", help="Path to model weights")
    parser.add_argument("--data", type=str, default="/tmp/sonaris_p1_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to evaluate")
    parser.add_argument("--output", type=str, default="reports/baseline_test_evaluation.json", help="Path to save output JSON")

    args = parser.parse_args()
    evaluate(model_path=args.model, data_yaml=args.data, split=args.split, output_json=args.output)


if __name__ == "__main__":
    main()
