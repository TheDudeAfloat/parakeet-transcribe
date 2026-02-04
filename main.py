from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import nemo.collections.asr as nemo_asr
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer
import torch
import shutil
import os
import subprocess
import tempfile

app = FastAPI()

# Configuration for documentation: 0.6b-v2 TDT model chosen for speed & zero-hallucination
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
model = None
inverse_normalizer = None

@app.on_event("startup")
async def load_model():
    global model, inverse_normalizer
    print(f"Loading {MODEL_NAME}...")
    # This should load from cache since we pre-downloaded it in the Dockerfile
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading InverseNormalizer...")
    inverse_normalizer = InverseNormalizer(lang='en')
    print("Model and Normalizer loaded and ready.")

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
        # Parakeet-TDT returns a list of Hypothesis objects
        results = model.transcribe([output_path])
        
        # 1. Extract raw text from the first hypothesis result
        # Check if the result is a Hypothesis object or a plain string (older models)
        hyp = results[0]
        raw_text = hyp.text if hasattr(hyp, 'text') else str(hyp)
        
        # 2. Apply Inverse Normalization (e.g., "one hundred" -> "100")
        if inverse_normalizer and raw_text:
            final_text = inverse_normalizer.inverse_normalize(raw_text, verbose=False)
        else:
            final_text = raw_text
        
        # 3. Return clean JSON (Drop-in compatibility for Speeches.ai)
        return {"text": final_text}
