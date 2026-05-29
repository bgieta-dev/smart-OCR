# Smart-OCR: Vision VLM Architecture (Qwen3-VL-4B)

Projekt wysokoprecyzyjnego systemu OCR do odczytu odręcznych danych tabelarycznych, zoptymalizowany pod karty graficzne z **8GB VRAM**.

## 🚀 Klucz do sukcesu (Perfect Accuracy)

Aby osiągnąć 100% dokładności przy odczycie gęstych tabel z pismem odręcznym, zastosowano następujące filary inżynieryjne:

### 1. Optymalizacja VRAM (The Goldilocks Config)
Dla modelu **Qwen3-VL-4B (4-bit AWQ/Compressed)** na karcie 8GB, kluczowe są parametry startowe vLLM:
*   `--max-model-len 2560`: Zbalansowany kontekst (1700 tokenów na obraz + zapas na prompt i odpowiedź).
*   `--gpu-memory-utilization 0.92`: Maksymalne wykorzystanie dostępnej pamięci.
*   `--enforce-eager`: Wyłączenie CUDA Graphs (oszczędność ~1GB VRAM).

### 2. Strategia "Surgical Slicing"
Przesyłanie całego arkusza do LLM powoduje halucynacje. System tnie obraz na mniejsze, odizolowane paski (slice'y) skupione wokół etykiet (P4, P13, P22).
*   **Izolacja**: Każdy slice zawiera tylko jedną tabelę, co eliminuje szum z sąsiednich sekcji.
*   **Dynamiczne Markery**: Wykorzystanie detekcji wzorca (kropek) do centrowania cięcia.

### 3. Structural Transcription (Markdown Strategy)
Zamiast prosić o listę liczb, wymuszamy na AI transkrypcję do **tabeli Markdown 10x6**.
*   **Spatial Discipline**: Model musi zdecydować o zawartości każdej z 60 kratek (liczba lub `-`). To zapobiega gubieniu liczb i zmyślaniu indeksów.
*   **English Prompting**: Użycie technicznego języka angielskiego w prompcie drastycznie poprawia "rozumowanie wizyjne" modelu.

### 4. Filtry Anty-Halucynacyjne (Backend Safeguards)
Nawet najlepszy model może zmyślać. `ai.py` zawiera twarde filtry:
*   **Entropy Filter**: Odrzucanie bloków o niskiej zmienności (np. ciągi `30, 35, 30...`).
*   **Label Filter**: Automatyczne usuwanie numeru etykiety z wyników.
*   **Pattern Detection**: Wykrywanie i ucinanie powtarzających się bloków danych.

## 🛠️ Uruchomienie

### Worker (Laptop z GPU)
```bash
./run_worker.sh
```

### Backend (CPU Server)
```bash
./run_app.sh
```

## 📝 Specyfikacja Modelu
*   **Model**: `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit`
*   **Format wag**: `compressed-tensors` (automatycznie wykrywany przez vLLM).
