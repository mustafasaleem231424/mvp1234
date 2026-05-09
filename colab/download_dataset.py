"""
Dataset Download & Preparation Script
=====================================
Downloads apple disease datasets and prepares them in YOLO format.

Supports two dataset sources:
1. PlantVillage (classification images → converted to YOLO detection format)
2. Roboflow (pre-annotated detection dataset)

Classes:
  0: apple_healthy
  1: apple_scab
  2: apple_black_rot
  3: apple_cedar_rust
  4: apple_frog_eye_spot
  5: bee_pollinator
"""

import os
import sys
import shutil
import random
import argparse
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import yaml

# ─── Configuration ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CLASS_NAMES = [
    "apple_healthy",
    "apple_scab",
    "apple_black_rot",
    "apple_cedar_rust",
    "apple_frog_eye_spot",
    "bee_pollinator",
]

CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# PlantVillage folder-to-class mapping
PLANTVILLAGE_MAPPING = {
    "Apple___healthy": "apple_healthy",
    "Apple___Apple_scab": "apple_scab",
    "Apple___Black_rot": "apple_black_rot",
    "Apple___Cedar_apple_rust": "apple_cedar_rust",
}

SPLIT_RATIOS = {"train": 0.70, "val": 0.20, "test": 0.10}


def download_plantvillage():
    """
    Download PlantVillage dataset.
    Since PlantVillage is a classification dataset (no bounding boxes),
    we create full-image bounding boxes for each leaf image.
    """
    print("\n" + "=" * 60)
    print("  PlantVillage Dataset Download")
    print("=" * 60)

    plantvillage_dir = RAW_DIR / "plantvillage"
    plantvillage_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if any(plantvillage_dir.iterdir()):
        print(f"✓ PlantVillage data found at {plantvillage_dir}")
        return plantvillage_dir

    print("\n⚠ PlantVillage dataset needs to be downloaded manually.")
    print("  Please follow these steps:\n")
    print("  Option A — Kaggle:")
    print("    1. Visit: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print("    2. Download the dataset ZIP")
    print(f"    3. Extract Apple folders to: {plantvillage_dir}")
    print()
    print("  Option B — Roboflow (recommended, pre-annotated):")
    print("    1. Visit: https://universe.roboflow.com/")
    print("    2. Search 'apple leaf disease detection'")
    print("    3. Export in YOLOv8 format")
    print(f"    4. Extract to: {plantvillage_dir}")
    print()
    print("  Option C — Use the Roboflow Python API:")
    print("    Run: python scripts/download_dataset.py --roboflow YOUR_API_KEY")
    print()

    return plantvillage_dir


def download_roboflow_dataset(api_key: str, workspace: str = "plant-disease-detection",
                               project: str = "apple-leaf-disease-detection",
                               version: int = 1):
    """
    Download dataset directly from Roboflow using API.
    """
    print("\n" + "=" * 60)
    print("  Roboflow Dataset Download")
    print("=" * 60)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    dataset = proj.version(version).download("yolov8", location=str(RAW_DIR / "roboflow"))

    print(f"✓ Dataset downloaded to: {dataset.location}")
    return Path(dataset.location)


def create_yolo_label_from_classification(image_path: Path, class_id: int, label_dir: Path):
    """
    For classification datasets (like PlantVillage) that don't have bounding boxes,
    create a YOLO label with a full-image bounding box.
    
    This assumes the entire image contains the leaf/disease.
    Format: class_id x_center y_center width height (all normalized 0-1)
    """
    label_path = label_dir / (image_path.stem + ".txt")

    # Full-image bounding box: center=(0.5, 0.5), size=(1.0, 1.0)
    # Slightly inset to 0.9 to avoid edge artifacts
    with open(label_path, "w") as f:
        f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")

    return label_path


def prepare_plantvillage_data(raw_dir: Path):
    """
    Convert PlantVillage classification structure to YOLO detection format.
    """
    print("\n📦 Converting PlantVillage to YOLO format...")

    all_samples = []  # List of (image_path, class_id)

    for folder_name, class_name in PLANTVILLAGE_MAPPING.items():
        folder_path = raw_dir / folder_name
        if not folder_path.exists():
            # Try alternate naming conventions
            alt_names = [
                folder_name.replace("___", " - "),
                folder_name.replace("___", "_"),
                class_name,
            ]
            for alt in alt_names:
                alt_path = raw_dir / alt
                if alt_path.exists():
                    folder_path = alt_path
                    break

        if not folder_path.exists():
            print(f"  ⚠ Folder not found: {folder_name} (skipping)")
            continue

        class_id = CLASS_MAP[class_name]
        images = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.JPG")) + \
                 list(folder_path.glob("*.png")) + list(folder_path.glob("*.PNG"))

        print(f"  ✓ {class_name}: {len(images)} images (class_id={class_id})")
        for img in images:
            all_samples.append((img, class_id))

    return all_samples


def prepare_bee_data(raw_dir: Path):
    """
    Prepare bee/pollinator images.
    Users should place bee images in data/raw/bee_pollinator/
    """
    bee_dir = raw_dir / "bee_pollinator"
    bee_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    class_id = CLASS_MAP["bee_pollinator"]

    images = list(bee_dir.glob("*.jpg")) + list(bee_dir.glob("*.JPG")) + \
             list(bee_dir.glob("*.png")) + list(bee_dir.glob("*.PNG"))

    if images:
        print(f"  ✓ bee_pollinator: {len(images)} images (class_id={class_id})")
        for img in images:
            all_samples.append((img, class_id))
    else:
        print(f"  ⚠ No bee images found in {bee_dir}")
        print(f"    Add bee/pollinator images to enable bee detection.")

    return all_samples


def split_and_copy(samples: list, use_classification_labels: bool = True):
    """
    Split samples into train/val/test and copy to processed directory.
    """
    print(f"\n📁 Splitting {len(samples)} samples...")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(samples)

    n = len(samples)
    train_end = int(n * SPLIT_RATIOS["train"])
    val_end = train_end + int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:],
    }

    for split_name, split_samples in splits.items():
        img_dir = PROCESSED_DIR / split_name / "images"
        lbl_dir = PROCESSED_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, class_id in split_samples:
            # Copy image
            dst_img = img_dir / img_path.name
            # Handle duplicate filenames
            counter = 1
            while dst_img.exists():
                dst_img = img_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
                counter += 1

            shutil.copy2(img_path, dst_img)

            # Create YOLO label
            if use_classification_labels:
                create_yolo_label_from_classification(dst_img, class_id, lbl_dir)

        print(f"  ✓ {split_name}: {len(split_samples)} samples")


def generate_data_yaml():
    """
    Generate the data.yaml configuration file for YOLO training.
    """
    data_config = {
        "path": str(PROCESSED_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ data.yaml saved to: {yaml_path}")
    print(f"  Classes ({len(CLASS_NAMES)}):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {i}: {name}")

    return yaml_path


def verify_dataset():
    """
    Verify the dataset structure and integrity.
    """
    print("\n" + "=" * 60)
    print("  Dataset Verification")
    print("=" * 60)

    all_ok = True

    for split in ["train", "val", "test"]:
        img_dir = PROCESSED_DIR / split / "images"
        lbl_dir = PROCESSED_DIR / split / "labels"

        images = list(img_dir.glob("*.*")) if img_dir.exists() else []
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        status = "✓" if not missing_labels and not orphan_labels and len(images) > 0 else "✗"
        print(f"\n  {status} {split}/")
        print(f"      Images: {len(images)}")
        print(f"      Labels: {len(labels)}")

        if missing_labels:
            print(f"      ⚠ Missing labels: {len(missing_labels)}")
            all_ok = False
        if orphan_labels:
            print(f"      ⚠ Orphan labels: {len(orphan_labels)}")
            all_ok = False
        if len(images) == 0:
            print(f"      ⚠ No images found!")
            all_ok = False

    # Check data.yaml
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"\n  ✓ data.yaml exists")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        print(f"      Classes: {config.get('nc', '?')}")
        print(f"      Names: {config.get('names', '?')}")
    else:
        print(f"\n  ✗ data.yaml missing!")
        all_ok = False

    if all_ok:
        print("\n✅ Dataset verification PASSED!")
    else:
        print("\n❌ Dataset verification FAILED — see warnings above.")

    return all_ok


def create_sample_dataset():
    """
    Create a small sample dataset with synthetic images for testing the pipeline.
    This allows you to verify the entire training pipeline works before
    downloading the full dataset.
    """
    print("\n" + "=" * 60)
    print("  Creating Sample Dataset (for pipeline testing)")
    print("=" * 60)

    sample_count = 20  # Per class
    img_size = (640, 640)

    for class_name in CLASS_NAMES:
        class_id = CLASS_MAP[class_name]

        for i in range(sample_count):
            # Create a simple colored image with some variation
            import numpy as np
            
            # Different base colors per class
            color_bases = {
                0: (34, 139, 34),    # Green for healthy
                1: (139, 90, 43),    # Brown for scab
                2: (20, 20, 20),     # Dark for black rot
                3: (210, 105, 30),   # Orange for cedar rust
                4: (128, 128, 0),    # Olive for frog eye
                5: (218, 165, 32),   # Gold for bee
            }

            base = color_bases.get(class_id, (100, 100, 100))
            # Add random noise for variation
            img_array = np.full((*img_size, 3), base, dtype=np.uint8)
            noise = np.random.randint(-30, 30, img_array.shape, dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array, "RGB")

            # Determine split
            if i < int(sample_count * 0.7):
                split = "train"
            elif i < int(sample_count * 0.9):
                split = "val"
            else:
                split = "test"

            img_dir = PROCESSED_DIR / split / "images"
            lbl_dir = PROCESSED_DIR / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{class_name}_{i:04d}"
            img.save(img_dir / f"{filename}.jpg")

            # Create YOLO label
            with open(lbl_dir / f"{filename}.txt", "w") as f:
                # Random bounding box within the image
                cx = random.uniform(0.3, 0.7)
                cy = random.uniform(0.3, 0.7)
                w = random.uniform(0.2, 0.5)
                h = random.uniform(0.2, 0.5)
                f.write(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    total = sample_count * len(CLASS_NAMES)
    print(f"  ✓ Created {total} sample images across {len(CLASS_NAMES)} classes")


# ─── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Download and prepare apple disease dataset")
    parser.add_argument("--roboflow", type=str, help="Roboflow API key for dataset download")
    parser.add_argument("--roboflow-workspace", type=str, default="plant-disease-detection",
                       help="Roboflow workspace name")
    parser.add_argument("--roboflow-project", type=str, default="apple-leaf-disease-detection",
                       help="Roboflow project name")
    parser.add_argument("--sample", action="store_true",
                       help="Create a small sample dataset for pipeline testing")
    parser.add_argument("--verify", action="store_true",
                       help="Verify existing dataset structure only")
    args = parser.parse_args()

    print("\n🍎 Apple Tree Disease Detection — Dataset Preparation")
    print("=" * 60)

    if args.verify:
        verify_dataset()
        return

    if args.sample:
        create_sample_dataset()
        generate_data_yaml()
        verify_dataset()
        return

    if args.roboflow:
        # Download from Roboflow API
        download_roboflow_dataset(
            api_key=args.roboflow,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
        )
        generate_data_yaml()
        verify_dataset()
        return

    # Default: prepare from local PlantVillage data
    plantvillage_dir = download_plantvillage()
    all_samples = prepare_plantvillage_data(plantvillage_dir)
    bee_samples = prepare_bee_data(RAW_DIR)
    all_samples.extend(bee_samples)

    if all_samples:
        split_and_copy(all_samples, use_classification_labels=True)
        generate_data_yaml()
        verify_dataset()
    else:
        print("\n⚠ No data found. Creating sample dataset for pipeline testing...")
        create_sample_dataset()
        generate_data_yaml()
        verify_dataset()


if __name__ == "__main__":
    main()
