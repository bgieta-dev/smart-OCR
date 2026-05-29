#!/bin/bash

# run_worker.sh - Wrapper for Vision VLM Architecture
# This script manages the GPU-accelerated containers on the Laptop.

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "--- Starting Smart-OCR Vision Worker (Qwen3-VL-4B) ---"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "Error: docker-compose or 'docker compose' is not installed."
    exit 1
fi

# Stop existing containers if running
echo "Cleaning up old containers..."
docker compose -f docker-compose.worker.yml down

# Start the Vision stack
echo "Starting vLLM Vision Server and OCR Worker..."
docker compose -f docker-compose.worker.yml up --build -d

echo ""
echo "--- DEPLOYMENT STATUS ---"
echo "Containers started in background."
echo ""
echo "To follow the LLM loading progress, run:"
echo "  docker logs -f vllm-vision"
echo ""
echo "To follow the Worker API logs, run:"
echo "  docker logs -f ocr-vision-worker"
echo ""
echo "NOTE: It may take several minutes to download the model and capture CUDA graphs."
echo "Wait until vLLM logs say 'Uvicorn running on http://0.0.0.0:8000' before using the app."
