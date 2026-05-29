import requests
import json
import os

# Laptop IP
WORKER_URL = "http://100.82.148.29:5000/process"

def send_to_laptop_worker(image_path, prompt):
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}
        
    with open(image_path, 'rb') as f:
        files = {'image': f}
        data = {'prompt': prompt}
        response = requests.post(WORKER_URL, files=files, data=data)
        
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Worker error: {response.text}"}

if __name__ == "__main__":
    # Example usage from server
    prompt = "Convert to JSON grid with keys like P1, P2 etc."
    result = send_to_laptop_worker("canvas.png", prompt)
    print(json.dumps(result, indent=2))
