import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass

import anyio
import nemo.collections.asr as nemo_asr
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

app = FastAPI()

# Configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v2")
MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", "8"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
TRANSCRIBE_CONCURRENCY = int(os.getenv("TRANSCRIBE_CONCURRENCY", "1"))
DEFAULT_FFMPEG_FILTERS = (
    "highpass=f=120,lowpass=f=3600,"
    "acompressor=threshold=-24dB:ratio=4:attack=5:release=100,"
    "loudnorm=I=-16:LRA=11:TP=-1.5"
)
FFMPEG_FILTERS = os.getenv("FFMPEG_FILTERS", DEFAULT_FFMPEG_FILTERS)
DISABLE_FILTERS = os.getenv("DISABLE_FILTERS", "false").lower() in {"1", "true", "yes", "on"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

model = None
inverse_normalizer = None
task_queue: asyncio.Queue | None = None
worker_task: asyncio.Task | None = None
transcribe_semaphore = asyncio.Semaphore(TRANSCRIBE_CONCURRENCY)


@dataclass
class TranscriptionTask:
    input_path: str
    output_path: str
    future: asyncio.Future


def _build_ffmpeg_command(input_path: str, output_path: str) -> list[str]:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
    ]

    if not DISABLE_FILTERS and FFMPEG_FILTERS:
        args += ["-af", FFMPEG_FILTERS]

    args.append(output_path)
    return args


def _run_ffmpeg(input_path: str, output_path: str) -> None:
    cmd = _build_ffmpeg_command(input_path, output_path)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logging.error("ffmpeg failed: %s", exc.stderr.strip())
        raise HTTPException(status_code=400, detail="Audio processing failed") from exc


def _transcribe_file(output_path: str) -> str:
    # Run the model transcription in a worker thread and respect concurrency guard
    results = model.transcribe([output_path])
    hyp = results[0]
    raw_text = hyp.text if hasattr(hyp, "text") else str(hyp)

    if inverse_normalizer and raw_text:
        try:
            return inverse_normalizer.inverse_normalize(raw_text, verbose=False)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Inverse normalization failed: %s", exc)
            return raw_text

    return raw_text


async def transcription_worker() -> None:
    assert task_queue is not None  # For type checkers

    while True:
        task: TranscriptionTask = await task_queue.get()
        try:
            if task.future.cancelled():
                continue

            preprocess_start = time.perf_counter()
            await anyio.to_thread.run_sync(_run_ffmpeg, task.input_path, task.output_path)
            logging.info(
                "Preprocessing complete in %.2fs", time.perf_counter() - preprocess_start
            )

            transcribe_start = time.perf_counter()
            async with transcribe_semaphore:
                final_text = await anyio.to_thread.run_sync(_transcribe_file, task.output_path)
            logging.info(
                "Transcription complete in %.2fs",
                time.perf_counter() - transcribe_start,
            )

            if not task.future.cancelled() and not task.future.done():
                task.future.set_result({"text": final_text})
        except Exception as exc:  # noqa: BLE001
            if not task.future.cancelled() and not task.future.done():
                task.future.set_exception(exc)
            logging.exception("Transcription task failed")
        finally:
            task_queue.task_done()

@app.on_event("startup")
async def load_model():
    global model, inverse_normalizer, task_queue, worker_task

    logging.info("Loading %s...", MODEL_NAME)
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading InverseNormalizer...")
    inverse_normalizer = InverseNormalizer(lang="en")
    logging.info("Model and Normalizer loaded and ready.")

    if task_queue is None:
        task_queue = asyncio.Queue(maxsize=MAX_PENDING_REQUESTS)

    if worker_task is None:
        worker_task = asyncio.create_task(transcription_worker())
        logging.info("Background transcription worker started.")


@app.on_event("shutdown")
async def shutdown_worker() -> None:
    global worker_task
    if worker_task:
        worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task

@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_NAME if model else "loading"}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model_name: str = Form(None), # Compatibility
    prompt: str = Form(None)      # Explicitly ignored to prevent prompt-bleeding
):
    if task_queue is None:
        raise HTTPException(status_code=503, detail="Service unavailable")

    tmpdir = tempfile.mkdtemp()
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    # Keep original extension if present to help ffmpeg, but avoid trusting the name itself
    extension = os.path.splitext(file.filename or "")[1][:10]
    safe_input = f"{uuid.uuid4().hex}{extension if extension else ''}"
    input_path = os.path.join(tmpdir, safe_input)
    output_path = os.path.join(tmpdir, f"{uuid.uuid4().hex}_resampled.wav")

    def _cleanup(_=None):
        shutil.rmtree(tmpdir, ignore_errors=True)

    future.add_done_callback(_cleanup)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        task = TranscriptionTask(input_path=input_path, output_path=output_path, future=future)

        if task_queue.full():
            _cleanup()
            raise HTTPException(status_code=429, detail="Too many pending requests. Please retry.")

        task_queue.put_nowait(task)
        logging.info(
            "Enqueued transcription task. Queue depth: %d/%d",
            task_queue.qsize(),
            task_queue.maxsize,
        )

        try:
            return await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise HTTPException(status_code=504, detail="Transcription timed out") from exc
    finally:
        file.file.close()
        if not future.done():
            # Ensure we don't leak temp directories if enqueue failed
            _cleanup()
