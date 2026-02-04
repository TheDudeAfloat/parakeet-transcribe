FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
# ffmpeg is required for audio conversion
# libsndfile1 is often required by audio libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the model weights
# We copy only the download script first so that this layer is cached
# unless the script or requirements change. This prevents re-downloading
# the large model every time main.py is modified.
COPY download_model.py .
RUN python download_model.py

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

