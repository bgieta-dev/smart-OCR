# Smart-OCR: Vision VLM Architecture (Qwen3-VL-4B)

Wysokoprecyzyjny system OCR zoptymalizowany do odczytu gęstych tabel z pismem odręcznym (10x6), dostosowany do pracy na kartach graficznych z **8GB VRAM**.

## 🚀 Filar Inżynieryjny (Perfect Accuracy)

System osiąga najwyższą precyzję dzięki połączeniu widzenia komputerowego (CV) i najnowszych modeli Vision-Language (VLM):

### 1. Strategia Micro-Slicing (Row-by-Row)
Zamiast analizować całą tabelę naraz (co przy 60 kratkach powoduje halucynacje w modelach 4B), system dzieli każdą sekcję na **6 poziomych mikrowierszy**.
*   **Asymetryczny Overlap**: Każdy wiersz jest wycinany z precyzyjnym marginesem (`-5px` góra, `-15px` dół), co zapobiega "widzeniu" sąsiednich rzędów i ucinaniu dolnych krawędzi cyfr.
*   **Spatial Discipline**: Model skupia się tylko na 10 liczbach naraz, co eliminuje błędy orientacji przestrzennej.

### 2. Dynamiczna Geometria (OpenCV Line Detection)
Górna sekcja arkusza (P4) często "pływa" na skanie. System używa **filtracji morfologicznej OpenCV**, aby:
*   Fizycznie zlokalizować pierwszą grubą linię tabeli.
*   Zignorować czarne kwadraty (markery) i nagłówki.
*   Dynamicznie "zaparkować" siatkę tnącą dokładnie na danych.

### 3. Architektura Multi-Node & Failover
System jest odporny na awarie i inteligentnie zarządza zasobami GPU:
*   **Auto-Health Check**: Przed rozpoczęciem pracy backend pinguje serwer główny (Remote). Jeśli jest offline, automatycznie przełącza się na laptopa (Backup).
*   **Zero-Disk Policy**: Cały proces (od PDF do JSON) odbywa się w **pamięci RAM**. Żadne obrazy nie są zapisywane na dysku, co zapewnia szybkość i bezpieczeństwo danych.

### 4. Optymalizacja Modelu (vLLM)
Wykorzystujemy model `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit` z parametrami:
*   `max-model-len 3200`: Zapas na wizualne tokeny o wysokiej rozdzielczości (300 DPI).
*   `temperature 0.0` & `penalty 0.0`: Ustawienia pozwalające na naturalne powtórzenia liczb przy zachowaniu pełnego determinizmu.

## 🛠️ Administracja i Debugowanie

### Przełącznik Debugowania
W pliku `ocr_config.py` znajduje się zmienna **`DEBUG_MODE`**:
*   `False` (Produkcja): Cicha praca, brak logów w terminalu, brak zapisu plików JPG.
*   `True` (Serwis): Generowanie map wizualnych `debug_map_section_X.jpg` z naniesionymi ramkami cięcia.

### Uruchomienie
1.  **Worker (GPU)**: `./run_worker.sh`
2.  **Backend (API)**: `./run_app.sh`

## 📝 Format Wynikowy
System zwraca czysty JSON, gdzie klucze zawierają pełne metadane sekcji (np. `P15 (2,40 dł)`), a wartości to listy zweryfikowanych, 2-cyfrowych pomiarów.
