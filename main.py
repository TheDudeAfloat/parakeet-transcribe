from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import nemo.collections.asr as nemo_asr
import torch
import shutil
import os
import subprocess
import tempfile

app = FastAPI()

# Configuration for documentation: 1.1b TDT model chosen for speed & zero-hallucination
#MODEL_NAME = "nvidia/parakeet-tdt-1.1b"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
model = None

@app.on_event("startup")
async def load_model():
    global model
    print(f"Loading {MODEL_NAME}...")
    # This should load from cache since we pre-downloaded it in the Dockerfile
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
