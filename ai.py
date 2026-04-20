import numpy as np
import os
from ollama import Client
import time
from PIL import Image
import io
import config
import json

# Pobieramy ścieżkę do katalogu, w którym znajduje się skrypt
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
    # Jeśli host nie został podany, używamy zdalnego z configu (domyślny)
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

        # Konwersja tablicy numpy na bajty JPEG dla Ollama (bez zapisu na dysku)
        img_pil = Image.fromarray(img_array)
        if img_pil.mode != 'RGB' and img_pil.mode != 'L':
             # Upewniamy się, że format jest kompatybilny
             img_pil = img_pil.convert('L')
             
        img_bytes_io = io.BytesIO()
        img_pil.save(img_bytes_io, format='JPEG')
        image_bytes = img_bytes_io.getvalue()

        start_total = time.time()
        prompt = prompt_reader()
        model = config.model
        
        # Próba połączenia z wybranym hostem
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
        try:
            response = call_ollama(host)
        except Exception as conn_err:
            if host == config.host_remote:
                print(f"Primary host {host} failed. Trying backup host {config.host_backup}...")
                try:
                    response = call_ollama(config.host_backup)
                except Exception as backup_err:
                    error_msg = f"Błąd połączenia: {str(backup_err)}"
                    return json.dumps({"error": error_msg})
            else:
                error_msg = f"Błąd połączenia: {str(conn_err)}"
                return json.dumps({"error": error_msg})
        
        print(f"AI Time: {time.time() - start_ai:.2f}s")
        print(f"Total Time: {time.time() - start_total:.2f}s")
        
        return response.message.content
    except Exception as e:
        return json.dumps({"error": f"Wystąpił błąd podczas przetwarzania AI: {str(e)}"})

if __name__ == "__main__":
    from image_processing import image_processing
    
    # Przetwarzamy obraz do np.array
    processed_array = image_processing('Skan 1.pdf')
    
    if processed_array is not None:
        # Przekazujemy np.array bezpośrednio do ai_check
        content = ai_check(processed_array)
        print(content)
    else:
        print("Błąd: image_processing nie zwrócił danych.")
