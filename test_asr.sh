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
