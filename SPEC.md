# Project Specification: Parakeet-ASR Drop-in API

## 1. Project Overview

* **Goal:** Create a Dockerized REST API that serves as a drop-in replacement for `speeches.ai` (OpenAI-compatible) using NVIDIA's **Parakeet-TDT-1.1b** model via the NeMo toolkit.
* **Target Use Case:** Transcribing narrow-band public safety radio traffic (8kHz) with high accuracy and zero "prompt-bleeding" hallucinations.
* **Primary Motivation:** Eliminate the "Whisper Looping" hallucination where the model repeats the prompt or previous sentences during static/silence.

## 2. Technical Stack

* **Base Image:** `nvcr.io/nvidia/pytorch:24.07-py3` (Required for optimized FastConformer kernels).
* **Frameworks:** FastAPI, Uvicorn, NVIDIA NeMo (ASR Toolkit).
* **Language:** Python 3.11+.
* **Dependencies:** `ffmpeg` (system), `libsndfile1` (system), `nemo_toolkit[asr]`, `python-multipart`.

## 3. Functional Requirements

### 3.1 API Compatibility

The API must accept a `POST` request to `/v1/audio/transcriptions` and support:

* `multipart/form-data` uploads.
* Parameters: `file` (audio binary), `model` (string, ignored), `prompt` (string, ignored to prevent hallucinations).
* **Response Format:** A JSON object: `{"text": "Extracted transcription here"}`.

### 3.2 Audio Preprocessing

* The system must accept `.m4a`, `.mp3`, and `.wav` formats.
* **Transformation:** All incoming audio must be processed with the following `ffmpeg` chain before inference:
    * **Format:** Resampled to **16,000Hz, Mono, 16-bit PCM**.
    * **Filters:**
        * Highpass: `f=120` (Remove rumble).
        * Lowpass: `f=3600` (Remove high-freq noise/hiss).
        * Compression: `threshold=-24dB:ratio=4` (Stabilize dynamic range).
        * Loudness: `loudnorm=I=-16:LRA=11:TP=-1.5` (Normalize volume).
* **Input Context:** Note that source audio is often 8kHz (narrow-band radio). The resampling logic must handle upscaling without introducing artifacts.

### 3.3 Model Selection

* **Default Model:** `nvidia/parakeet-tdt-1.1b` (Retrieved from Hugging Face, no NGC Key required).
* **Compute:** Must utilize `cuda` for inference.

## 4. Implementation Details

### 4.1 Directory Structure

```text
/
├── Dockerfile          # Multi-stage build preferred
├── main.py            # FastAPI application logic
├── requirements.txt    # Python dependencies
└── SPEC.md            # This document

```

### 4.2 Key Logic Constraints

* **Concurrency:** Use `tempfile` for audio processing to prevent file collisions during simultaneous transcription requests.
* **Cleanliness:** Ensure all temporary audio files are deleted immediately after the transcription is returned.
* **Documentation:** All technical configurations that depart from default (e.g., custom ffmpeg flags) must be commented in the source.

## 5. Acceptance Criteria

1. **Connectivity:** A `cURL` request from the host machine to the container returns a valid JSON response.
2. **Performance:** Transcription of a 30-second 8kHz radio call should take < 500ms on an NVIDIA GPU.
3. **Accuracy:** The system must not output any text from the `prompt` parameter if the audio contains only static.
