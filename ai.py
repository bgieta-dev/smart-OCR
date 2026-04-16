import numpy as np
import os
from ollama import Client
import time
from PIL import Image
import io
import config
# Pobieramy ścieżkę do katalogu, w którym znajduje się skrypt
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def prompt_reader():
    prompt_path = os.path.join(SCRIPT_DIR, "prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"Ostrzeżenie: Nie znaleziono pliku promptu: {prompt_path}")
        return "Dont overthink - just read numbers and then list them in json format for each P<number>"
    with open(prompt_path, "r") as f:
        return f.read().strip()

def ai_check(img_path, host=None):
    # Jeśli host nie został podany, używamy zdalnego z configu (domyślny)
    if host is None:
        host = config.host_remote

    # Jeśli ścieżka nie jest absolutna, szukaj jej względem katalogu skryptu
    if not os.path.isabs(img_path):
        img_path = os.path.join(SCRIPT_DIR, img_path)

    if not os.path.exists(img_path):
        return f"Błąd: Nie znaleziono pliku obrazu: {img_path}"

    try:
        img = Image.open(img_path).convert('L')  
        img_array = np.array(img)
        model = config.model
        img_pil = Image.fromarray(img_array, mode='L')
        img_bytes = io.BytesIO()
        img_pil.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Get the actual bytes from BytesIO
        image_bytes = img_bytes.getvalue()

        start_total = time.time()
        prompt = prompt_reader()
        print(f"Sending to Ollama ({host}) (Using {model}) image: {os.path.basename(img_path)}...")
        start_ai = time.time()
        
        # Inicjalizacja klienta Ollama z konkretnym hostem
        client = Client(host=host)
        
        response = client.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': "Extract all numbers from all P-number sections in the image and return them in the specified JSON format.", 'images': [image_bytes]}
            ],
            format='json',
            options={
                'temperature': 0.0,
            },
        )
        
        print(f"AI Time: {time.time() - start_ai:.2f}s")
        print(f"Total Time: {time.time() - start_total:.2f}s")
        
        content = response.message.content
        return content
    except Exception as e:
        return f"Wystąpił błąd podczas przetwarzania AI: {str(e)}"

if __name__ == "__main__":
    content = ai_check("ready_image.png")
    print(content)
