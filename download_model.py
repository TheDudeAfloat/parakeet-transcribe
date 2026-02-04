import nemo.collections.asr as nemo_asr
import os

# Configuration matching main.py
MODEL_NAME = "nvidia/parakeet-tdt-1.1b"

print(f"Pre-loading {MODEL_NAME} for Docker caching...")

# This triggers the download and caching
# NeMo caches models by default, usually in ~/.cache/torch/NeMo/
# We instantiate it to ensure it's fully downloaded.
try:
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    print(f"Model {MODEL_NAME} downloaded successfully.")
except Exception as e:
    print(f"Failed to download model: {e}")
    exit(1)
