"""
Data Augmentation Pipeline
===========================
Advanced augmentation for apple disease images using Albumentations.
Generates augmented copies to expand the training dataset.
"""

import os
import argparse
import random
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import albumentations as A
    from albumentations.core.serialization import save as save_transform
except ImportError:
    print("Installing albumentations...")
    os.system("pip install albumentations")
    import albumentations as A


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def get_augmentation_pipeline(severity: str = "medium"):
    """
    Create augmentation pipeline based on severity level.
    
    Args:
        severity: 'light', 'medium', or 'heavy'
    """
    if severity == "light":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.5
            ),
            A.Rotate(limit=15, p=0.3),
        ])

    elif severity == "medium":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.25,
                contrast_limit=0.25,
                p=0.6
            ),
            A.Rotate(limit=30, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.4
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=5,
                p=0.2
            ),
        ])

    else:  # heavy
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.35,
                contrast_limit=0.35,
                p=0.7
            ),
            A.Rotate(limit=45, p=0.6),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.3),
            A.RandomShadow(
                shadow_roi=(0, 0.3, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=0.3
            ),
            A.CLAHE(clip_limit=4.0, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
        ])


def augment_image(image_path: Path, label_path: Path, output_img_dir: Path,
                  output_lbl_dir: Path, transform: A.Compose, num_augmented: int = 3):
    """
    Generate augmented copies of an image and its corresponding YOLO label.
    
    For YOLO labels, since we're doing geometric + color transforms on classification-style
    bounding boxes (full image), we keep the label as-is. For actual bounding box datasets,
    you'd need to apply bbox-aware transforms.
    """
    try:
        img = np.array(Image.open(image_path).convert("RGB"))
    except Exception as e:
        print(f"  ⚠ Failed to load {image_path}: {e}")
        return 0

    # Read original label
    label_content = ""
    if label_path.exists():
        with open(label_path) as f:
            label_content = f.read()

    count = 0
    for i in range(num_augmented):
        augmented = transform(image=img)
        aug_img = Image.fromarray(augmented["image"])

        # Save augmented image
        aug_name = f"{image_path.stem}_aug{i:02d}{image_path.suffix}"
        aug_img.save(output_img_dir / aug_name)

        # Copy label (unchanged for full-image bboxes)
        aug_label_name = f"{image_path.stem}_aug{i:02d}.txt"
        with open(output_lbl_dir / aug_label_name, "w") as f:
            f.write(label_content)

        count += 1

    return count


def augment_dataset(split: str = "train", severity: str = "medium",
                    num_augmented: int = 3, target_class: str = None):
    """
    Augment all images in a split.
    
    Args:
        split: 'train', 'val', or 'test'
        severity: augmentation intensity
        num_augmented: copies per image
        target_class: optionally augment only a specific class
    """
    img_dir = PROCESSED_DIR / split / "images"
    lbl_dir = PROCESSED_DIR / split / "labels"

    if not img_dir.exists():
        print(f"⚠ Directory not found: {img_dir}")
        return

    transform = get_augmentation_pipeline(severity)
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    if target_class:
        images = [img for img in images if target_class in img.stem]

    print(f"\n🔄 Augmenting {len(images)} images in {split}/ (severity={severity})")
    print(f"   Generating {num_augmented} copies per image...")

    total_generated = 0
    for idx, img_path in enumerate(images):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        count = augment_image(
            img_path, lbl_path,
            img_dir, lbl_dir,
            transform, num_augmented
        )
        total_generated += count

        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(images)} images...")

    print(f"\n✓ Generated {total_generated} augmented images")
    print(f"  Total images in {split}/images: {len(list(img_dir.glob('*.*')))}")


def main():
    parser = argparse.ArgumentParser(description="Augment training data")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "val", "test"],
                       help="Dataset split to augment")
    parser.add_argument("--severity", type=str, default="medium",
                       choices=["light", "medium", "heavy"],
                       help="Augmentation severity level")
    parser.add_argument("--num", type=int, default=3,
                       help="Number of augmented copies per image")
    parser.add_argument("--target-class", type=str, default=None,
                       help="Only augment a specific class (e.g., 'apple_scab')")
    args = parser.parse_args()

    print("\n🍎 Apple Disease Detection — Data Augmentation")
    print("=" * 60)

    augment_dataset(
        split=args.split,
        severity=args.severity,
        num_augmented=args.num,
        target_class=args.target_class
    )


if __name__ == "__main__":
    main()
