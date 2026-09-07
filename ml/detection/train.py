"""
Sonaris P1: AI/ML Detection Module
Baseline YOLO Model Training Pipeline
"""

import argparse
import json
import shutil
import time
from pathlib import Path
import torch
from ultralytics import YOLO


def get_default_device():
    """Detect available device with priority: MPS (Apple Silicon) -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def train_model(
    data="/tmp/sonaris_p1_dataset/data.yaml",
    model="yolo11n.pt",
    epochs=25,
    batch=16,
    imgsz=640,
    device=None,
    project="models/baseline",
    name="run",
    seed=42,
    patience=10,
    save_period=5,
    export_dir="models/weights",
):
    """
    Train a YOLO model on the Sonaris dataset with reproducibility and validation.

    Args:
        data (str): Path to dataset YAML configuration.
        model (str): Pretrained YOLO weights or architecture YAML.
        epochs (int): Number of training epochs.
        batch (int): Batch size.
        imgsz (int): Image resolution for training.
        device (str): Compute device ('mps', 'cuda:0', 'cpu'). Auto-detected if None.
        project (str): Project output directory for runs.
        name (str): Experiment run name.
        seed (int): Random seed for reproducibility.
        patience (int): Early stopping patience epochs.
        save_period (int): Save checkpoint every N epochs.
        export_dir (str): Destination directory for the best model weights.

    Returns:
        dict: Training summary including paths, metrics, and duration.
    """
    if device is None:
        device = get_default_device()

    data_path = Path(data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {data_path}")

    print("=" * 60)
    print("SONARIS P1: BASELINE YOLO TRAINING")
    print("=" * 60)
    print(f"Dataset config   : {data_path}")
    print(f"Base model       : {model}")
    print(f"Epochs           : {epochs}")
    print(f"Batch size       : {batch}")
    print(f"Image size       : {imgsz}")
    print(f"Compute device   : {device}")
    print(f"Reproducibility  : seed={seed}, deterministic=True")
    print(f"Project output   : {project}/{name}")
    print("=" * 60)

    start_time = time.time()

    # Initialize model
    yolo_model = YOLO(model)

    # Launch training
    results = yolo_model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        seed=seed,
        deterministic=True,
        patience=patience,
        save_period=save_period,
        plots=True,
        verbose=True,
    )

    duration_sec = time.time() - start_time
    duration_str = f"{duration_sec / 60:.2f} minutes ({duration_sec:.1f}s)"

    # Identify output weight paths
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    best_weights_src = save_dir / "weights" / "best.pt"
    last_weights_src = save_dir / "weights" / "last.pt"

    export_path = Path(export_dir).resolve()
    export_path.mkdir(parents=True, exist_ok=True)
    best_weights_dest = export_path / "best.pt"

    if best_weights_src.exists():
        shutil.copy2(best_weights_src, best_weights_dest)
        print(f"\nSuccessfully exported best weights to: {best_weights_dest}")
    else:
        print(f"\nWarning: {best_weights_src} not found; falling back to {last_weights_src}")
        if last_weights_src.exists():
            shutil.copy2(last_weights_src, best_weights_dest)

    # Extract validation metrics from training summary
    val_metrics = {}
    if hasattr(results, "results_dict") and results.results_dict:
        val_metrics = {
            "val_precision": float(results.results_dict.get("metrics/precision(B)", 0.0)),
            "val_recall": float(results.results_dict.get("metrics/recall(B)", 0.0)),
            "val_mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0.0)),
            "val_mAP50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)),
        }

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "data": str(data_path),
        "epochs_requested": epochs,
        "batch_size": batch,
        "image_size": imgsz,
        "device": str(device),
        "seed": seed,
        "training_duration_seconds": round(duration_sec, 2),
        "training_duration_formatted": duration_str,
        "run_directory": str(save_dir),
        "best_weights_source": str(best_weights_src),
        "best_weights_export": str(best_weights_dest),
        "best_val_metrics": val_metrics,
    }

    # Save training record metadata
    config_record_path = save_dir / "train_config.json"
    with open(config_record_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved training configuration record to: {config_record_path}")

    # Also save in models root for quick reference
    models_record_path = Path("models/train_config.json")
    models_record_path.parent.mkdir(parents=True, exist_ok=True)
    with open(models_record_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE in {duration_str}")
    if val_metrics:
        print(f"Best Val mAP50    : {val_metrics.get('val_mAP50', 0.0):.4f}")
        print(f"Best Val mAP50-95 : {val_metrics.get('val_mAP50_95', 0.0):.4f}")
    print(f"Weights Exported  : {best_weights_dest}")
    print("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on Sonaris dataset")
    parser.add_argument("--data", type=str, default="/tmp/sonaris_p1_dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Initial model weights")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('mps', '0', 'cpu')")
    parser.add_argument("--project", type=str, default="models/baseline", help="Project dir")
    parser.add_argument("--name", type=str, default="yolo11n_baseline", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--export-dir", type=str, default="models/weights", help="Directory to export best.pt")
    parser.add_argument("--sanity-check", action="store_true", help="Run 1 epoch sanity check only")

    args = parser.parse_args()

    epochs = 1 if args.sanity_check else args.epochs
    run_name = f"{args.name}_sanity" if args.sanity_check else args.name

    train_model(
        data=args.data,
        model=args.model,
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=run_name,
        seed=args.seed,
        patience=args.patience,
        export_dir=args.export_dir,
    )


if __name__ == "__main__":
    main()
