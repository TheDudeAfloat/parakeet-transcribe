
# Parakeet setup

---

### 1. The `docker-compose.yml`

This configuration specifically requests the NVIDIA GPU and sets up a health check to ensure the model is loaded before you start sending traffic.

```yaml
services:
  parakeet-asr:
    build: .
    container_name: parakeet-asr-api
    restart: always
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONUNBUFFERED=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s  # Gives the 1.1b model time to load into VRAM

    # Volume mount if you want to debug audio files locally
    volumes:
      - ./temp_audio:/tmp/asr_processing

```

---

### 2. The Final Implementation Wrapper (`main.py`)

This version adds a `/health` endpoint and handles the 8kHz upsampling you mentioned earlier using the `ffmpeg` logic.

```python
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import nemo.collections.asr as nemo_asr
import torch
import shutil
import os
import subprocess
import tempfile

app = FastAPI()

# Configuration for documentation: 1.1b TDT model chosen for speed & zero-hallucination
MODEL_NAME = "nvidia/parakeet-tdt-1.1b"
model = None

@app.on_event("startup")
async def load_model():
    global model
    print(f"Loading {MODEL_NAME}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print("Model loaded and ready.")

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME if model else "loading"}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(None), # Compatibility
    prompt: str = Form(None)      # Explicitly ignored to prevent prompt-bleeding
):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        output_path = os.path.join(tmpdir, "resampled.wav")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Resample any source (8kHz, 44.1kHz, etc.) to 16kHz Mono for Parakeet
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', input_path,
                '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                output_path
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=400, detail="Audio processing failed")

        # Inference
        transcriptions = model.transcribe([output_path])
        return {"text": transcriptions[0]}

```

---

### 3. The Bash Test Script (`test_asr.sh`)

This script will help you verify the drop-in compatibility immediately.

```bash
#!/bin/bash
# test_asr.sh

API_URL="http://localhost:8000/v1/audio/transcriptions"
FILENAME=$1
PROMPT="HCF 1A DISP - Engine 1, Medic 5" # This will be ignored by Parakeet

if [ -z "$FILENAME" ]; then
    echo "Usage: ./test_asr.sh path/to/audio.m4a"
    exit 1
fi

echo "Transcribing $FILENAME..."
curl -sS -X POST "$API_URL" \
     -H "Accept: application/json" \
     -F "file=@${FILENAME}" \
     -F "prompt=${PROMPT}" \
     -F "response_format=json" | jq .

```
