"""
YOLOv8 Fine-Tuning Script
===========================
Fine-tunes a pre-trained YOLOv8 model on the apple disease dataset.

Usage:
    python scripts/train.py                          # Train with defaults
    python scripts/train.py --model yolov8s.pt       # Use small model
    python scripts/train.py --epochs 50 --batch 8    # Custom params
    python scripts/train.py --resume                 # Resume training
"""

import argparse
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_training_config():
    """Load training hyperparameters from config file."""
    config_path = PROJECT_ROOT / "configs" / "model_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def train(args):
    """Run YOLOv8 fine-tuning."""
    from ultralytics import YOLO

    # Load config
    config = load_training_config()

    # Override config with CLI arguments
    model_name = args.model or config.get("model", "yolov8n.pt")
    epochs = args.epochs or config.get("epochs", 100)
    batch = args.batch or config.get("batch", 16)
    imgsz = args.imgsz or config.get("imgsz", 640)
    device = args.device if args.device is not None else config.get("device", 0)
    project = config.get("project", "runs/apple_disease")
    name = args.name or config.get("name", "v1")

    # Data config
    data_yaml = PROJECT_ROOT / "data" / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found! Run download_dataset.py first.")
        print(f"   Expected at: {data_yaml}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  🍎 Apple Disease Detection — YOLOv8 Training")
    print("=" * 60)
    print(f"\n  Model:       {model_name}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch}")
    print(f"  Image size:  {imgsz}")
    print(f"  Device:      {device}")
    print(f"  Data config: {data_yaml}")
    print(f"  Output:      {project}/{name}")
    print()

    # Load pre-trained model
    if args.resume:
        # Resume from last checkpoint
        last_weights = PROJECT_ROOT / project / name / "weights" / "last.pt"
        if last_weights.exists():
            print(f"📂 Resuming training from: {last_weights}")
            model = YOLO(str(last_weights))
        else:
            print(f"⚠ No checkpoint found at {last_weights}, starting fresh...")
            model = YOLO(model_name)
    else:
        print(f"📂 Loading pre-trained model: {model_name}")
        model = YOLO(model_name)

    # Training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "project": str(PROJECT_ROOT / project),
        "name": name,
        "patience": config.get("patience", 20),
        "save": config.get("save", True),
        "save_period": config.get("save_period", 10),
        "plots": config.get("plots", True),
        "cos_lr": config.get("cos_lr", True),
        "optimizer": config.get("optimizer", "AdamW"),
        "lr0": config.get("lr0", 0.01),
        "lrf": config.get("lrf", 0.001),
        "weight_decay": config.get("weight_decay", 0.0005),
        "warmup_epochs": config.get("warmup_epochs", 5),
        "warmup_momentum": config.get("warmup_momentum", 0.8),
        "mosaic": config.get("mosaic", 1.0),
        "mixup": config.get("mixup", 0.15),
        "copy_paste": config.get("copy_paste", 0.1),
        "hsv_h": config.get("hsv_h", 0.015),
        "hsv_s": config.get("hsv_s", 0.7),
        "hsv_v": config.get("hsv_v", 0.4),
        "degrees": config.get("degrees", 15.0),
        "translate": config.get("translate", 0.1),
        "scale": config.get("scale", 0.5),
        "shear": config.get("shear", 2.0),
        "flipud": config.get("flipud", 0.5),
        "fliplr": config.get("fliplr", 0.5),
        "verbose": True,
        "exist_ok": True,  # Don't increment run name
    }

    if args.resume:
        train_args["resume"] = True

    # Start training
    print("\n🚀 Starting training...\n")
    results = model.train(**train_args)

    # Print results summary
    print("\n" + "=" * 60)
    print("  ✅ Training Complete!")
    print("=" * 60)

    best_weights = PROJECT_ROOT / project / name / "weights" / "best.pt"
    print(f"\n  Best weights: {best_weights}")
    print(f"  Results dir:  {PROJECT_ROOT / project / name}")
    print()

    # Quick validation
    print("📊 Running final validation...")
    val_results = model.val()
    print(f"\n  mAP@0.5:     {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95: {val_results.box.map:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for apple disease detection")
    parser.add_argument("--model", type=str, default=None,
                       help="Model to use (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                       help="Image size for training")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (0, 1, cpu)")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training from last checkpoint")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
