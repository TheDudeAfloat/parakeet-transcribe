# Parakeet-ASR Drop-in API

A high-performance, GPU-accelerated Automatic Speech Recognition (ASR) API using NVIDIA's **Parakeet-TDT** models. Designed as a drop-in replacement for `speeches.ai` (OpenAI-compatible) for transcribing narrow-band public safety radio traffic with high accuracy and zero hallucinations.

## Key Features

-   **Zero "Prompt-Bleeding":** Eliminates hallucinations where the model repeats the prompt during silence.
-   **Inverse Normalization:** Automatically converts number words to digits (e.g., "four seventeen" -> "417").
-   **Optimized for Radio:** Includes `ffmpeg` preprocessing to upsample 8kHz narrow-band audio to 16kHz Mono 16-bit PCM.
-   **Dockerized:** Ready-to-deploy container with NVIDIA GPU support and model caching.
-   **Configurable:** Easily switch between Parakeet models (e.g., `0.6b-v2` for speed, `1.1b` for accuracy).

## Prerequisites

-   **NVIDIA GPU** (Tested on RTX 4070)
-   **Docker** & **Docker Compose**
-   **NVIDIA Container Toolkit** (Required for GPU access in Docker)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ginoguidi/parakeet-transcribe.git
    cd parakeet-transcribe
    ```

2.  **Configure environment:**
    The project comes with a default `.env` file. You can modify the model version here:
    ```bash
    # .env
    MODEL_NAME=nvidia/parakeet-tdt-0.6b-v2
    PORT=8007
    ```

3.  **Build and Run:**
    ```bash
    docker-compose up --build -d
    ```
    *Note: The first build will take some time as it downloads the ~2GB model weights to cache them in the Docker image.*

4.  **Verify Health:**
    ```bash
    curl http://localhost:8007/health
    # {"status": "healthy", "model": "nvidia/parakeet-tdt-0.6b-v2"}
    ```

## Usage

### API Endpoint

**POST** `/v1/audio/transcriptions`

Accepts `multipart/form-data` uploads. Compatible with OpenAI client libraries (ignoring unsupported parameters).

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `file` | Binary | The audio file (wav, mp3, m4a). |
| `model` | String | Ignored (Compatibility). |
| `prompt` | String | **Ignored** to prevent hallucinations. |

### Example Request

```bash
curl -X POST "http://localhost:8007/v1/audio/transcriptions" \
     -F "file=@/path/to/audio.wav"
```

### Response

```json
{
  "text": "Engine 1 responding to 417 Main Street."
}
```

## Configuration

| Environment Variable | Default |
| :--- | :--- |
| `MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v2` | The NeMo ASR model to load. |
| `PORT` | `8007` | The external port exposed by Docker. |

## Troubleshooting

-   **OOM Errors:** If you encounter Out-Of-Memory errors, ensure your `docker-compose.yml` has `shm_size: '2gb'` and `PYTORCH_CUDA_ALLOC_CONF` set (already included in default config).
-   **Slow Startup:** The container downloads the model on the first build. Subsequent restarts are instant.

## License

MIT
