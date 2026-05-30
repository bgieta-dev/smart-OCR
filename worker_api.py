from flask import Flask, request, jsonify
import base64
import os
from openai import OpenAI

app = Flask(__name__)

# --- CONFIG ---
LLM_MODEL = os.environ.get("LLM_MODEL", "cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")


# --- INIT LLM CLIENT ---
client = OpenAI(api_key="token-ignored", base_url=VLLM_URL)

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    prompt = request.form.get('prompt', "Describe this image.")
    
    if os.environ.get("DEBUG_MODE", "False").lower() == "true":
        print("\n--- WORKER (VISION): RECEIVED NEW TASK ---")
    
    # Read image and convert to base64
    img_data = image_file.read()
    base64_image = base64.b64encode(img_data).decode('utf-8')
    
    data = request.form
    prompt = data.get("prompt", "Analyze this image.")
    # Add a unique salt to the prompt to bypass prefix caching
    request_id = data.get("request_id", "0")
    prompt = f"{prompt}\n\n[Request ID: {request_id}]"

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            max_tokens=800,
            temperature=0.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<EOF>", "Row 7", "Row 8"],
            seed=int(request_id) if request_id.isdigit() else 42
        )

        structured = response.choices[0].message.content
        if os.environ.get("DEBUG_MODE", "False").lower() == "true":
            print(f"WORKER VISION RESULT:\n{structured}")
        
        return jsonify({
            "raw_text": "Vision processing (no raw text stage)",
            "structured": structured
        })
    except Exception as e:
        print(f"WORKER ERROR: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
