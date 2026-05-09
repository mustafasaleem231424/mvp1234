import os
import sys
import shutil
from pathlib import Path
from PIL import Image

def main():
    print("🚀 Generating local model for the web app...")
    
    # The full 38 PlantVillage classes expected by our web app.
    # Must be in this exact alphabetical order so class indices match the frontend.
    classes = [
        'Apple___Apple_scab',
        'Apple___Black_rot',
        'Apple___Cedar_apple_rust',
        'Apple___healthy',
        'Blueberry___healthy',
        'Cherry_(including_sour)___Powdery_mildew',
        'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
        'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy',
        'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)',
        'Peach___Bacterial_spot',
        'Peach___healthy',
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight',
        'Potato___healthy',
        'Raspberry___healthy',
        'Soybean___healthy',
        'Squash___Powdery_mildew',
        'Strawberry___Leaf_scorch',
        'Strawberry___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    # 1. Create a tiny dummy dataset
    data_dir = Path('dummy_dataset')
    if data_dir.exists():
        shutil.rmtree(data_dir)
        
    for split in ['train', 'val']:
        for cls in classes:
            dir_path = data_dir / split / cls
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create a couple of 224x224 RGB dummy images for each class
            for i in range(2):
                img = Image.new('RGB', (224, 224), color=(73, 109, 137))
                img.save(dir_path / f"dummy_{i}.jpg")
                
    print("✅ Created dummy dataset.")
    
    # 2. Train for 1 epoch to bake classes into the model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics not installed yet. Please wait for pip install to finish.")
        sys.exit(1)
        
    model = YOLO('yolov8n-cls.pt')
    
    print("🏃 Training lightweight local model...")
    model.train(
        data=str(data_dir),
        epochs=1,
        imgsz=224,
        batch=2,
        project='runs',
        name='dummy_run',
        exist_ok=True,
        verbose=False
    )
    
    # 3. Export to TFJS
    print("📦 Exporting to TensorFlow.js...")
    best_weights = Path('runs/dummy_run/weights/best.pt')
    
    # Export to saved_model first
    export_model = YOLO(best_weights)
    export_model.export(format='saved_model', imgsz=224)
    
    # Convert to TFJS
    saved_model_dir = Path('runs/dummy_run/weights/best_saved_model')
    tfjs_dir = Path('public/model')
    
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)
        
    os.system(f"tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model '{saved_model_dir}' '{tfjs_dir}'")
    
    print("🎉 Done! The model is now available at public/model/model.json")
    print("You can run 'npm run dev' to see the app working.")

if __name__ == '__main__':
    main()
