# Smart-OCR: Hybrid Pipeline (EasyOCR + LLM)

System do precyzyjnej ekstrakcji danych z tabel 6x10 przy użyciu hybrydowego podejścia:
1. **EasyOCR** (Stage 1) - Wykrywa surowy tekst i cyfry z obrazu.
2. **Qwen2.5 LLM** (Stage 2) - Naprawia błędy OCR i układa dane w ustrukturyzowany JSON.

## Wymagania sprzętowe i uruchomienie

Wybierz wersję zależnie od swojej karty graficznej (VRAM):

### Opcja A: Karta 16GB+ (np. RX 7900, RX 9070 XT)
Używa pełnego modelu `Qwen2.5-7B` dla najwyższej inteligencji.
```bash
docker compose -f docker-compose.vllm.yml up -d
```

### Opcja B: Karta 8GB AMD (np. RX 6600, RX 7600)
Używa modelu skwantyzowanego `Qwen2.5-7B-AWQ` pod ROCm.
```bash
docker compose -f docker-compose.vllm.8gb.yml up -d
```

### Opcja C: Karta 8GB NVIDIA (np. RTX 4060 Laptop)
Używa modelu skwantyzowanego `Qwen2.5-7B-AWQ` pod CUDA. 
*Wymaga zainstalowanego `nvidia-container-toolkit` na systemie hosta.*
```bash
docker compose -f docker-compose.vllm.nvidia.8gb.yml up -d
```

## Konfiguracja i Instalacja

1. **Python Dependencies:**
   Upewnij się, że masz zainstalowane biblioteki (w tym PyTorch z obsługą ROCm dla AMD):
   ```bash
   pip install -r requirements.txt
   ```

2. **ocr_config.py:**
   Jeśli używasz wersji 8GB, upewnij się, że w `ocr_config.py` nazwa modelu zgadza się z tą w pliku `.yml`:
   * Dla 16GB: `model = "Qwen/Qwen2.5-7B-Instruct"`
   * Dla 8GB: `model = "Qwen/Qwen2.5-7B-Instruct-AWQ"`

## Uruchomienie Procesu
```bash
python ai.py
```

## Architektura i Debugowanie
- Skrypt tnie obraz wejściowy na 3 sekcje na podstawie markerów (`debug_slice_*.jpg`).
- EasyOCR przetwarza każdą sekcję na tekst.
- LLM (vLLM server) naprawia błędy (np. zamienia 'O' na '0') i zwraca finalny JSON.
- Wszystkie kontenery mają ustawione `restart: always`, więc będą wstawać razem z systemem.
