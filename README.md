# Smart-OCR

Narzędzie do ekstrakcji danych numerycznych z dokumentów PDF i obrazów przy użyciu sztucznej inteligencji.

## Funkcjonalność

Program przetwarza skany dokumentów i ekstrahuje liczby z oznaczonych sekcji (P<number>) zwracając dane w uporządkowanej strukturze JSON.

- Wsparcie dla plików PDF i obrazów
- Automatyczna orientacja dokumentu
- Wykorzystanie AI (Ollama + qwen3.5:4b) do rozpoznawania OCR
- Wyniki w formacie JSON

## Instalacja

```bash
pip install -r requirements.txt
```

## Użycie

```bash
python3 ai.py
```

## Wymagania

- Python 3.7+
- Ollama z modelem qwen3.5:4b
- OpenCV
- PIL/Pillow
- pdf2image
- ollama-python

## Struktura projektu

- `ai.py` - główny moduł AI
- `image_processing.py` - przetwarzanie obrazów
- `prompt.txt` - prompt dla AI
- `template.png` - szablon do dopasowania
