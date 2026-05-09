"""
=============================================================
🌿 Plant Disease Detection — YOLOv8 Classification Training
=============================================================
Google Colab Training Script — Target: 96%+ Accuracy

This script trains a YOLOv8 classification model on the PlantVillage
dataset (54,000+ images, 38 classes, 14 crop species).

USAGE IN GOOGLE COLAB:
  1. Upload this file to Colab or clone the repo
  2. Set runtime to GPU (Runtime → Change runtime type → T4 GPU)
  3. Run: !python train_plant_disease.py

The script will:
  - Install dependencies
  - Download the PlantVillage dataset from Kaggle
  - Organize it into train/val/test splits
  - Train YOLOv8m-cls with optimized hyperparameters
  - Evaluate and export the model for TensorFlow.js
=============================================================
"""

import os
import sys
import shutil
import random
import json
from pathlib import Path
from datetime import datetime

# ─── Step 0: Install Dependencies ────────────────────────────
def install_dependencies():
    """Install all required packages for Colab environment."""
    print("\n" + "=" * 60)
    print("  Step 0: Installing Dependencies")
    print("=" * 60)
    
    packages = [
        "ultralytics>=8.2.0",
        "albumentations>=1.3.0",
        "tensorflowjs>=4.0.0",   # For model export
        "scikit-learn>=1.3.0",
        "seaborn>=0.12.0",
        "onnx>=1.14.0",          # For ONNX export step
    ]
    for pkg in packages:
        os.system(f"{sys.executable} -m pip install -q {pkg}")
    
    print("✅ Dependencies installed")


# ─── Step 1: Download Dataset ────────────────────────────────
def download_dataset(data_root: Path):
    """
    Download PlantVillage dataset.
    
    Strategy: Use Hugging Face datasets library (no Kaggle API key needed).
    Falls back to a direct download if HF is unavailable.
    """
    print("\n" + "=" * 60)
    print("  Step 1: Downloading PlantVillage Dataset")
    print("=" * 60)
    
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if (raw_dir / "plantvillage").exists() and len(list((raw_dir / "plantvillage").rglob("*.jpg"))) > 1000:
        count = len(list((raw_dir / "plantvillage").rglob("*.jpg")))
        print(f"✅ Dataset already downloaded ({count} images)")
        return raw_dir / "plantvillage"
    
    # Method 1: Try Kaggle (if API key is available)
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "kaggle.json").exists():
        print("📦 Downloading from Kaggle...")
        os.system("pip install -q kaggle")
        os.system(
            f"kaggle datasets download -d emmarex/plantdisease "
            f"-p {raw_dir} --unzip"
        )
        # Find the extracted directory
        for candidate in ["PlantVillage", "plantvillage", "New Plant Diseases Dataset(Augmented)"]:
            candidate_path = raw_dir / candidate
            if candidate_path.exists():
                target = raw_dir / "plantvillage"
                if candidate_path != target:
                    shutil.move(str(candidate_path), str(target))
                print(f"✅ Downloaded from Kaggle")
                return target
    
    # Method 2: Hugging Face (no API key needed)
    print("📦 Downloading from Hugging Face...")
    try:
        os.system(f"{sys.executable} -m pip install -q datasets Pillow")
        from datasets import load_dataset
        
        ds = load_dataset("mohanty/PlantVillage", split="train")
        
        plantvillage_dir = raw_dir / "plantvillage"
        
        # Get label names
        label_names = ds.features["label"].names
        print(f"   Found {len(label_names)} classes, {len(ds)} images")
        
        # Save images organized by class
        for idx, sample in enumerate(ds):
            label_name = label_names[sample["label"]]
            # Clean label name for filesystem
            clean_name = label_name.replace(" ", "_").replace("___", "__")
            class_dir = plantvillage_dir / clean_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            img = sample["image"]
            img.save(class_dir / f"{idx:06d}.jpg")
            
            if (idx + 1) % 5000 == 0:
                print(f"   Saved {idx + 1}/{len(ds)} images...")
        
        total = len(list(plantvillage_dir.rglob("*.jpg")))
        print(f"✅ Downloaded {total} images in {len(label_names)} classes")
        return plantvillage_dir
        
    except Exception as e:
        print(f"⚠️ Hugging Face download failed: {e}")
        print("\n📋 Manual download instructions:")
        print("   1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease")
        print("   2. Download and unzip to: data/raw/plantvillage/")
        print("   3. Re-run this script")
        sys.exit(1)


# ─── Step 2: Organize Dataset ────────────────────────────────
def organize_dataset(raw_dir: Path, data_root: Path, val_ratio=0.15, test_ratio=0.10):
    """
    Organize the raw dataset into train/val/test splits.
    
    YOLOv8 classification expects:
      dataset/
        train/
          class_name/
            image1.jpg
        val/
          class_name/
            image1.jpg
        test/
          class_name/
            image1.jpg
    
    Args:
        raw_dir: Path to raw PlantVillage directory
        data_root: Root data directory
        val_ratio: Fraction for validation (default 15%)
        test_ratio: Fraction for test (default 10%)
    """
    print("\n" + "=" * 60)
    print("  Step 2: Organizing Dataset (Train/Val/Test Split)")
    print("=" * 60)
    
    dataset_dir = data_root / "dataset"
    
    # Check if already organized
    train_dir = dataset_dir / "train"
    if train_dir.exists() and len(list(train_dir.iterdir())) > 5:
        n_train = len(list(train_dir.rglob("*.jpg")))
        print(f"✅ Dataset already organized ({n_train} training images)")
        return dataset_dir
    
    # Create split directories
    for split in ["train", "val", "test"]:
        (dataset_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Get all class directories
    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"   Found {len(class_dirs)} classes")
    
    random.seed(42)  # Reproducible splits
    
    total_stats = {"train": 0, "val": 0, "test": 0}
    class_stats = {}
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + \
                 list(class_dir.glob("*.png")) + list(class_dir.glob("*.PNG"))
        
        if len(images) == 0:
            continue
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n = len(images)
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test
        
        splits = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        
        for split_name, split_images in splits.items():
            dest_dir = dataset_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                shutil.copy2(img_path, dest_dir / img_path.name)
            
            total_stats[split_name] += len(split_images)
        
        class_stats[class_name] = {
            "total": n, "train": len(splits["train"]),
            "val": len(splits["val"]), "test": len(splits["test"]),
        }
    
    # Print summary
    print(f"\n   Dataset Split Summary:")
    print(f"   {'Split':<10} {'Images':>8}")
    print(f"   {'-'*20}")
    for split, count in total_stats.items():
        print(f"   {split:<10} {count:>8}")
    print(f"   {'-'*20}")
    print(f"   {'Total':<10} {sum(total_stats.values()):>8}")
    
    # Save class mapping
    class_names = sorted(class_stats.keys())
    mapping = {i: name for i, name in enumerate(class_names)}
    mapping_path = data_root / "class_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n   Class mapping saved to: {mapping_path}")
    print(f"   Total classes: {len(class_names)}")
    
    print(f"\n✅ Dataset organized into {dataset_dir}")
    return dataset_dir


# ─── Step 3: Train Model ─────────────────────────────────────
def train_model(dataset_dir: Path, project_dir: Path):
    """
    Fine-tune YOLOv8m-cls on the plant disease dataset.
    
    Strategy for 96%+ accuracy:
    - Use yolov8m-cls (medium) — best accuracy/speed tradeoff on Colab GPU
    - Strong data augmentation to prevent overfitting
    - Cosine LR schedule with warmup
    - Early stopping with patience=15
    - AdamW optimizer with weight decay
    - Image size 224 (standard for classification)
    """
    print("\n" + "=" * 60)
    print("  Step 3: Training YOLOv8m-cls Model")
    print("  Target: 96%+ Top-1 Accuracy")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    # Load pre-trained classification model
    # yolov8m-cls has ~12M params — good balance for Colab T4 GPU
    model = YOLO("yolov8m-cls.pt")
    
    print(f"\n   Model: YOLOv8m-cls (pre-trained on ImageNet)")
    print(f"   Dataset: {dataset_dir}")
    print(f"   Device: GPU (auto-detected)")
    
    # Training with optimized hyperparameters for high accuracy
    results = model.train(
        data=str(dataset_dir),
        epochs=80,              # 80 epochs with early stopping
        imgsz=224,              # Standard classification size
        batch=64,               # Larger batch for stable gradients (T4 has 16GB)
        
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,              # Initial learning rate (lower for fine-tuning)
        lrf=0.01,               # Final LR = lr0 * lrf = 0.00001
        weight_decay=0.0005,
        momentum=0.937,
        
        # Learning Rate Schedule
        cos_lr=True,            # Cosine annealing — smooth decay
        warmup_epochs=5,        # Warm up for first 5 epochs
        warmup_momentum=0.8,
        
        # Early Stopping
        patience=15,            # Stop if no improvement for 15 epochs
        
        # Data Augmentation (strong but not destructive)
        augment=True,
        hsv_h=0.015,            # Hue variation
        hsv_s=0.5,              # Saturation variation
        hsv_v=0.3,              # Brightness variation
        degrees=15.0,           # Rotation range
        translate=0.1,          # Translation
        scale=0.3,              # Zoom
        shear=2.0,              # Shear
        flipud=0.2,             # Vertical flip (leaves can be upside down)
        fliplr=0.5,             # Horizontal flip
        erasing=0.1,            # Random erasing (robustness)
        
        # Output
        project=str(project_dir),
        name="plant_disease_v1",
        exist_ok=True,
        save=True,
        save_period=10,         # Checkpoint every 10 epochs
        plots=True,             # Generate training plots
        verbose=True,
        
        # Performance
        workers=4,
        seed=42,                # Reproducible training
    )
    
    # Print final results
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    
    best_weights = project_dir / "plant_disease_v1" / "weights" / "best.pt"
    print(f"\n   Best weights: {best_weights}")
    
    return model, best_weights


# ─── Step 4: Evaluate Model ──────────────────────────────────
def evaluate_model(model, dataset_dir: Path, project_dir: Path):
    """
    Comprehensive evaluation with confusion matrix and per-class metrics.
    """
    print("\n" + "=" * 60)
    print("  Step 4: Model Evaluation")
    print("=" * 60)
    
    # Validate on val set
    print("\n📊 Validation set results:")
    val_results = model.val(
        data=str(dataset_dir),
        split="val",
        plots=True,
        verbose=True,
    )
    
    val_top1 = val_results.top1
    val_top5 = val_results.top5
    
    print(f"\n   Top-1 Accuracy: {val_top1 * 100:.2f}%")
    print(f"   Top-5 Accuracy: {val_top5 * 100:.2f}%")
    
    # Test set evaluation
    print("\n📊 Test set results:")
    test_results = model.val(
        data=str(dataset_dir),
        split="test",
        plots=True,
        verbose=True,
    )
    
    test_top1 = test_results.top1
    test_top5 = test_results.top5
    
    print(f"\n   Top-1 Accuracy: {test_top1 * 100:.2f}%")
    print(f"   Top-5 Accuracy: {test_top5 * 100:.2f}%")
    
    # Accuracy check
    if test_top1 >= 0.96:
        print(f"\n   🎯 TARGET MET! {test_top1*100:.2f}% >= 96%")
    else:
        print(f"\n   ⚠️ Below target: {test_top1*100:.2f}% < 96%")
        print("   Consider: more epochs, larger model (yolov8l-cls), or more augmentation")
    
    # Save results summary
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "model": "yolov8m-cls",
        "dataset": "PlantVillage",
        "validation": {"top1": float(val_top1), "top5": float(val_top5)},
        "test": {"top1": float(test_top1), "top5": float(test_top5)},
        "target_met": test_top1 >= 0.96,
    }
    
    results_path = project_dir / "plant_disease_v1" / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n   Results saved to: {results_path}")
    return results_summary


# ─── Step 5: Export Model ─────────────────────────────────────
def export_model(model, best_weights: Path, project_dir: Path):
    """
    Export the trained model in multiple formats:
    - ONNX (for cross-platform inference)
    - TorchScript (for PyTorch deployment)
    - TensorFlow SavedModel (for TF.js conversion)
    """
    print("\n" + "=" * 60)
    print("  Step 5: Exporting Model")
    print("=" * 60)
    
    from ultralytics import YOLO
    
    export_model = YOLO(str(best_weights))
    export_dir = project_dir / "plant_disease_v1" / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX (most compatible)
    print("\n📦 Exporting to ONNX...")
    try:
        export_model.export(format="onnx", imgsz=224, simplify=True)
        print("   ✅ ONNX export complete")
    except Exception as e:
        print(f"   ⚠️ ONNX export failed: {e}")
    
    # Export to TorchScript
    print("\n📦 Exporting to TorchScript...")
    try:
        export_model.export(format="torchscript", imgsz=224)
        print("   ✅ TorchScript export complete")
    except Exception as e:
        print(f"   ⚠️ TorchScript export failed: {e}")
    
    # Export to TensorFlow SavedModel (needed for TF.js)
    print("\n📦 Exporting to TensorFlow SavedModel...")
    try:
        export_model.export(format="saved_model", imgsz=224)
        print("   ✅ TF SavedModel export complete")
        
        # Convert to TensorFlow.js
        print("\n📦 Converting to TensorFlow.js...")
        saved_model_dir = best_weights.parent / best_weights.stem + "_saved_model"
        tfjs_dir = export_dir / "tfjs_model"
        
        os.system(
            f"tensorflowjs_converter --input_format=tf_saved_model "
            f"--output_format=tfjs_graph_model "
            f"'{saved_model_dir}' '{tfjs_dir}'"
        )
        
        if tfjs_dir.exists():
            print(f"   ✅ TF.js model saved to: {tfjs_dir}")
        else:
            print("   ⚠️ TF.js conversion may need manual steps")
            
    except Exception as e:
        print(f"   ⚠️ TF export failed: {e}")
        print("   You can manually convert later with:")
        print("   model.export(format='saved_model')")
    
    print(f"\n✅ All exports saved to: {export_dir}")


# ─── Step 6: Generate Report ─────────────────────────────────
def generate_report(results: dict, project_dir: Path):
    """Generate a human-readable training report."""
    print("\n" + "=" * 60)
    print("  Step 6: Training Report")
    print("=" * 60)
    
    report = f"""
╔══════════════════════════════════════════════════════════╗
║         🌿 Plant Disease Detection — Training Report     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Model:     YOLOv8m-cls (fine-tuned)                     ║
║  Dataset:   PlantVillage (54,000+ images, 38 classes)    ║
║  Species:   14 crop types                                ║
║  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}                         ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  RESULTS                                                 ║
╠──────────────────────────────────────────────────────────╣
║                                                          ║
║  Validation Top-1:  {results['validation']['top1']*100:6.2f}%                         ║
║  Validation Top-5:  {results['validation']['top5']*100:6.2f}%                         ║
║  Test Top-1:        {results['test']['top1']*100:6.2f}%                         ║
║  Test Top-5:        {results['test']['top5']*100:6.2f}%                         ║
║                                                          ║
║  Target (96%):      {'✅ MET' if results['target_met'] else '❌ NOT MET'}                              ║
║                                                          ║
╠══════════════════════════════════════════════════════════╣
║  FILES                                                   ║
╠──────────────────────────────────────────────────────────╣
║  Best weights:  plant_disease_v1/weights/best.pt         ║
║  Last weights:  plant_disease_v1/weights/last.pt         ║
║  ONNX model:    plant_disease_v1/exports/best.onnx       ║
║  Train plots:   plant_disease_v1/                        ║
╚══════════════════════════════════════════════════════════╝
"""
    print(report)
    
    # Save report to file
    report_path = project_dir / "plant_disease_v1" / "TRAINING_REPORT.txt"
    with open(report_path, "w") as f:
        f.write(report)


# ─── Main Pipeline ───────────────────────────────────────────
def main():
    """Run the complete training pipeline."""
    print("""
    ╔═══════════════════════════════════════════════╗
    ║  🌿 Plant Disease Detection Training Pipeline  ║
    ║  YOLOv8m-cls · PlantVillage · Target: 96%+   ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Paths
    BASE_DIR = Path.cwd()
    DATA_DIR = BASE_DIR / "data"
    PROJECT_DIR = BASE_DIR / "runs"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Pipeline
    install_dependencies()
    
    raw_dir = download_dataset(DATA_DIR)
    dataset_dir = organize_dataset(raw_dir, DATA_DIR)
    
    model, best_weights = train_model(dataset_dir, PROJECT_DIR)
    
    results = evaluate_model(model, dataset_dir, PROJECT_DIR)
    
    export_model(model, best_weights, PROJECT_DIR)
    
    generate_report(results, PROJECT_DIR)
    
    print("\n🎉 Pipeline complete! Check the runs/ directory for all outputs.")


if __name__ == "__main__":
    main()
