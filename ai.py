import numpy as np
from ollama import chat
from ollama import ChatResponse
import time
from PIL import Image
import io

def prompt_reader():
    with open("prompt.txt", "r") as f:
        return f.read().strip()

def ai_check(img_path):
    img = Image.open(img_path).convert('L')  
    img_array = np.array(img)
    
    img_pil = Image.fromarray(img_array, mode='L')
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Get the actual bytes from BytesIO
    image_bytes = img_bytes.getvalue()

    start_total = time.time()
    prompt = prompt_reader()
    print("Sending to Ollama (Using qwen3.5:4b)...")
    start_ai = time.time()
    response: ChatResponse = chat(
    model='qwen3.5:4b',
    messages=[
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': "Dont overthink - just read numbers and then list them in json format for each P<number>", 'images': [image_bytes]}
    ],
    format='json',
    options={
        'temperature': 0.9,
    },
    )
    
    print(f"AI Time: {time.time() - start_ai:.2f}s")
    print(f"Total Time: {time.time() - start_total:.2f}s")
    
    return response.message.content

if __name__ == "__main__":
    content = ai_check("ready_image.png")
    print(content)
