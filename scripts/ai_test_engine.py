import requests
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor

# Our local API endpoint
API_URL = "http://localhost:3000/api/analyze"

# Sample set of diverse plant disease images from the internet
# These are chosen to test the "Expert AI" range (Coffee, Wheat, Cocoa, etc.)
TEST_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Coffee_leaf_rust.jpg/1024px-Coffee_leaf_rust.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Wheat_stem_rust_on_wheat_ears.jpg/1024px-Wheat_stem_rust_on_wheat_ears.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Cassava_mosaic_disease.jpg/1024px-Cassava_mosaic_disease.jpg",
    "https://images.fineartamerica.com/images/artworkimages/mediumlarge/1/black-spot-on-rose-leaf-nigel-cattlin.jpg",
    "https://gardenerspath.com/wp-content/uploads/2021/05/Powdery-Mildew-on-Zucchini-Leaves.jpg",
    "https://extension.umn.edu/sites/extension.umn.edu/files/styles/large/public/apple-scab-leaf-lesions.jpg",
    "https://i0.wp.com/post.healthline.com/wp-content/uploads/2020/08/poison-ivy-leaf-1296x728-header.jpg", # Test for plant ID
]

def test_image(url):
    print(f"Testing: {url[:60]}...")
    try:
        # Download image
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"url": url, "error": "Failed to download image"}

        # Encode to base64
        encoded_string = base64.b64encode(response.content).decode('utf-8')
        image_data = f"data:image/jpeg;base64,{encoded_string}"

        # Call our API
        api_res = requests.post(API_URL, json={"image": image_data}, timeout=30)
        if api_res.status_code != 200:
            return {"url": url, "error": f"API Error: {api_res.text}"}

        result = api_res.json()
        if result["success"]:
            diag = result["result"]["topPrediction"]["diseaseInfo"]
            print(f"✅ SUCCESS: {diag['crop']} - {diag['disease']} ({result['result']['light']})")
            return {"url": url, "result": diag}
        else:
            return {"url": url, "error": result.get("error", "Unknown error")}

    except Exception as e:
        return {"url": url, "error": str(e)}

def run_mass_test(count=10):
    print(f"🚀 Starting Mass AI Diagnostic Test ({count} samples)...")
    results = []
    # We use a smaller loop here for demonstration, but it's built to scale
    for url in TEST_URLS[:count]:
        res = test_image(url)
        results.append(res)
        time.sleep(2) # Respect Gemini rate limits (15 RPM)
    
    with open("ai_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📝 Test complete. Results saved to ai_test_results.json")

if __name__ == "__main__":
    # Ensure the local dev server is running before executing this
    print("NOTE: Make sure 'npm run dev' is running in another terminal!")
    run_mass_test()
