"""
Local Model Tester
====================
Run this script AFTER downloading the trained 'best.pt' or 'best.onnx' model.
It will run inference on a sample image and print the results to verify
accuracy before integrating it into the web app.
"""

import sys
import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
    from PIL import Image
except ImportError:
    print("Missing requirements. Please run:")
    print("pip install ultralytics pillow")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test trained YOLOv8 classification model.")
    parser.add_argument("--model", type=str, default="public/model/best.pt", help="Path to best.pt or best.onnx")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    args = parser.parse_args()

    model_path = Path(args.model)
    img_path = Path(args.image)

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please place the trained best.pt file there first.")
        sys.exit(1)

    if not img_path.exists():
        print(f"❌ Image not found at {img_path}")
        sys.exit(1)

    print(f"🔄 Loading model from {model_path}...")
    model = YOLO(str(model_path))

    print(f"🔍 Analyzing {img_path.name}...")
    
    # Run prediction
    results = model.predict(str(img_path), verbose=False)
    result = results[0]

    # Get top 3 predictions
    top_indices = result.probs.top5[:3]  # Take top 3
    print("\n" + "="*40)
    print("  RESULTS")
    print("="*40)
    
    for i, idx in enumerate(top_indices):
        class_name = result.names[idx]
        confidence = result.probs.data[idx].item() * 100
        
        prefix = "🎯 TOP PREDICTION:" if i == 0 else "   Alternative:"
        print(f"{prefix:<20} {class_name:<25} ({confidence:.1f}%)")
    
    print("="*40)

if __name__ == "__main__":
    main()
