# Smart-OCR - Dokumentacja modułu

Narzędzie do inteligentnej ekstrakcji danych numerycznych z dokumentów (PDF/obrazy) przy użyciu lokalnego modelu AI (Ollama).

## Architektura i Moduły

- **`ai.py`**: Integracja z Ollama. Wysyła przetworzony obraz do modelu i oczekuje odpowiedzi w formacie JSON. Parametry (model, parametry temperatury itp.) są w `config.py`.
- **`image_processing.py`**: Przetwarzanie obrazu przed wysłaniem do AI. Konwersja PDF -> Image, wyrównanie obrazu przy użyciu OpenCV (Template Matching z `template.png`).
- **`prompt.txt`**: Kluczowy plik z instrukcjami systemowymi dla modelu AI. Zmiany w tym pliku bezpośrednio wpływają na jakość ekstrakcji.

## Tech Stack

- **AI**: Ollama (`qwen2.5:14b` lub `qwen3.5:4b`)
- **Obrazy**: `OpenCV` (cv2), `Pillow` (PIL), `pdf2image`
- **Model danych**: Zwracany JSON z polami takimi jak: `invoice_number`, `date`, `total_amount` itp.

## Uwagi dla Gemini CLI

- **Prompt Engineering**: Zawsze najpierw edytuj `prompt.txt`, jeśli AI nie wyciąga danych poprawnie.
- **Image Processing**: Funkcja `image_processing()` zwraca przetworzony obraz jako tablicę NumPy (`np.array`), która jest następnie przekazywana bezpośrednio do `ai_check()`. Moduł nie zapisuje już plików tymczasowych na dysku.
- **Ollama**: Pamiętaj, że serwer Ollama musi być dostępny lokalnie (zgodnie z `config.py`).
