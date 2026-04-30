import numpy as np
import os
from ollama import Client
import time
from PIL import Image
import io
import config
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def prompt_reader():
    prompt_path = os.path.join(SCRIPT_DIR, "prompt.txt")
    if not os.path.exists(prompt_path):
        print(f"Ostrzeżenie: Nie znaleziono pliku promptu: {prompt_path}")
        return "Dont overthink - just read numbers and then list them in json format for each P<number>"
    with open(prompt_path, "r") as f:
        return f.read().strip()

def ai_check(img_input, host=None):
    """
    img_input: Może być ścieżką do pliku (string) lub tablicą numpy (np.array)
    """
    if host is None:
        host = config.host_remote

    try:
        if isinstance(img_input, str):
            # Logika dla ścieżki do pliku
            if not os.path.isabs(img_input):
                img_path = os.path.join(SCRIPT_DIR, img_input)
            else:
                img_path = img_input

            if not os.path.exists(img_path):
                return json.dumps({"error": f"Błąd: Nie znaleziono pliku obrazu: {img_path}"})
            
            img = Image.open(img_path).convert('L')
            img_array = np.array(img)
            image_name = os.path.basename(img_path)
        elif isinstance(img_input, np.ndarray):
            # Logika dla tablicy numpy (to o co prosiłeś)
            img_array = img_input
            image_name = "numpy_memory_array"
        else:
            return json.dumps({"error": f"Błąd: Nieobsługiwany typ wejścia: {type(img_input)}"})

        img_pil = Image.fromarray(img_array)
        if img_pil.mode != 'RGB' and img_pil.mode != 'L':
             img_pil = img_pil.convert('L')
             
        img_bytes_io = io.BytesIO()
        img_pil.save(img_bytes_io, format='JPEG')
        image_bytes = img_bytes_io.getvalue()

        start_total = time.time()
        prompt = prompt_reader()
        model = config.model
        
        def call_ollama(target_host):
            print(f"Sending to Ollama ({target_host}) (Using {model}) source: {image_name}...")
            client = Client(host=target_host)
            return client.chat(
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

        start_ai = time.time()
        
        hosts_to_try = [host] if host else [config.host_remote, config.host_backup]
        
        response = None
        for current_host in hosts_to_try:
            try:
                response = call_ollama(current_host)
                break 
            except Exception as conn_err:
                print(f"Host {current_host} failed: {conn_err}")
                continue
        
        if response is None:
            return json.dumps({"error": "Błąd: Wszystkie hosty Ollama są nieosiągalne."})

        print(f"AI Time: {time.time() - start_ai:.2f}s")
        print(f"Total Time: {time.time() - start_total:.2f}s")
        
        return response.message.content
    except Exception as e:
        return json.dumps({"error": f"Wystąpił błąd podczas przetwarzania AI: {str(e)}"})

if __name__ == "__main__":
    from image_processing import image_processing
    
    try:
        # Przetwarzamy obraz do np.array
        processed_array = image_processing('Skan 1.pdf')
        
        # Przekazujemy np.array bezpośrednio do ai_check
        content = ai_check(processed_array)
        print(content)
    except Exception as e:
        print(f"Błąd: {e}")
