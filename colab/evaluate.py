"""
Model Evaluation Script
========================
Comprehensive evaluation of the trained YOLOv8 apple disease detection model.

Usage:
    python scripts/evaluate.py --weights runs/apple_disease/v1/weights/best.pt
    python scripts/evaluate.py --weights best.pt --source data/processed/test/images
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(args):
    """Run comprehensive model evaluation."""
    from ultralytics import YOLO
    import yaml

    weights_path = Path(args.weights)
    if not weights_path.exists():
        # Try relative to project root
        weights_path = PROJECT_ROOT / args.weights
    if not weights_path.exists():
        print(f"❌ Weights not found: {args.weights}")
        sys.exit(1)

    # Load data config
    data_yaml = PROJECT_ROOT / "data" / "data.yaml"
    if not data_yaml.exists():
        print("❌ data.yaml not found!")
        sys.exit(1)

    with open(data_yaml) as f:
        data_config = yaml.safe_load(f)

    class_names = data_config.get("names", [])

    print("\n" + "=" * 60)
    print("  🍎 Apple Disease Detection — Model Evaluation")
    print("=" * 60)
    print(f"\n  Weights:  {weights_path}")
    print(f"  Data:     {data_yaml}")
    print(f"  Classes:  {len(class_names)}")
    print()

    # Load model
    model = YOLO(str(weights_path))

    # ─── Validation on val set ───
    print("📊 Evaluating on validation set...")
    val_results = model.val(
        data=str(data_yaml),
        split="val",
        plots=True,
        save_json=True,
        verbose=True,
    )

    print("\n" + "─" * 60)
    print("  Validation Results")
    print("─" * 60)
    print(f"  mAP@0.5:       {val_results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {val_results.box.map:.4f}")
    print(f"  Precision:      {val_results.box.mp:.4f}")
    print(f"  Recall:         {val_results.box.mr:.4f}")

    # Per-class metrics
    print("\n  Per-Class Performance:")
    print("  " + "─" * 56)
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'mAP@0.5':>10}")
    print("  " + "─" * 56)

    for i, name in enumerate(class_names):
        if i < len(val_results.box.ap50):
            p = val_results.box.p[i] if i < len(val_results.box.p) else 0
            r = val_results.box.r[i] if i < len(val_results.box.r) else 0
            ap = val_results.box.ap50[i]
            print(f"  {name:<25} {p:>10.4f} {r:>10.4f} {ap:>10.4f}")

    # ─── Test set evaluation (if exists) ───
    test_dir = Path(data_config["path"]) / "test" / "images"
    if test_dir.exists() and any(test_dir.iterdir()):
        print("\n\n📊 Evaluating on test set...")
        test_results = model.val(
            data=str(data_yaml),
            split="test",
            plots=True,
            verbose=True,
        )

        print("\n" + "─" * 60)
        print("  Test Results")
        print("─" * 60)
        print(f"  mAP@0.5:       {test_results.box.map50:.4f}")
        print(f"  mAP@0.5:0.95:  {test_results.box.map:.4f}")

    # ─── Sample predictions ───
    if args.predict_samples:
        print("\n\n🖼 Running sample predictions...")
        sample_dir = test_dir if test_dir.exists() else Path(data_config["path"]) / "val" / "images"
        
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg"))[:args.num_samples]
            
            if sample_images:
                pred_results = model.predict(
                    source=sample_images,
                    save=True,
                    save_txt=True,
                    conf=args.conf,
                    project=str(PROJECT_ROOT / "runs" / "apple_disease"),
                    name="eval_predictions",
                    exist_ok=True,
                )

                print(f"\n  ✓ Sample predictions saved to: runs/apple_disease/eval_predictions/")
                
                for result in pred_results:
                    boxes = result.boxes
                    if len(boxes) > 0:
                        for box in boxes:
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
                            print(f"    → {Path(result.path).name}: {name} ({conf:.2f})")
                    else:
                        print(f"    → {Path(result.path).name}: No detections")

    # ─── Speed benchmark ───
    if args.benchmark:
        print("\n\n⚡ Speed Benchmark...")
        benchmark_results = model.benchmark(
            data=str(data_yaml),
            imgsz=640,
            half=False,
            device=args.device or "0",
        )
        print(f"\n  Benchmark results saved.")

    # ─── Export results ───
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "weights": str(weights_path),
        "metrics": {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr),
        },
        "per_class": {},
    }

    for i, name in enumerate(class_names):
        if i < len(val_results.box.ap50):
            results_summary["per_class"][name] = {
                "ap50": float(val_results.box.ap50[i]),
            }

    results_path = PROJECT_ROOT / "runs" / "apple_disease" / "evaluation_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n\n📄 Results saved to: {results_path}")
    print("\n✅ Evaluation complete!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv8 model")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to trained weights (best.pt)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold for predictions")
    parser.add_argument("--predict-samples", action="store_true", default=True,
                       help="Run sample predictions")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of sample predictions")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run speed benchmark")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use for evaluation")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
