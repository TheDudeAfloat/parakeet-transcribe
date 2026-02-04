#!/bin/bash

# Configuration
SPEACHES_URL="http://localhost:8000/v1/audio/transcriptions"
PARAKEET_URL="http://localhost:8007/v1/audio/transcriptions"
AUDIO_FILE=$1

# You mentioned using these in your radio scripts
PROMPT="his is San Diego public safety radio traffic, including police, fire, and EMS dispatch. The audio may mention units such as Engine 1, Engine 5, Engine 12, Medic 5, Truck 20, Battalion 1, and Patrol 3. When the speaker says an engine unit, use the word Engine (fire engine), not Agent. Other common terms include ALS, Palomar, 647F, 11-5, K-9, Able, Otay, hazmat, A-firm, check the welfare, Metro, 10-4, Yellow, Kaiser Zion, RP, Medic, Station A, Mercy, DV, Scripps La Jolla, Scripps Mercy, Training Delay, Failure to Yield, San Diego PD, Chula Vista, FTY, Sheriff, National City, Carlsbad, Oceanside, LE South, LE North, Santee, CHP, Engine."

if [ -z "$AUDIO_FILE" ]; then
    echo "Usage: ./compare_asr.sh <audio_file>"
    exit 1
fi

echo "--------------------------------------------------------"
echo "Testing File: $AUDIO_FILE"
echo "--------------------------------------------------------"

# 1. Test speaches.ai (Faster-Whisper)
echo "Sending to speaches.ai (Whisper)..."
start_time=$(date +%s.%N)
whisper_res=$(curl -sS -X POST "$SPEACHES_URL" \
    -H "Accept: application/json" \
    -F "file=@${AUDIO_FILE}" \
    -F "model=Systran/faster-whisper-large-v3" \
    -F "prompt=${PROMPT}" \
    -F "response_format=json")
end_time=$(date +%s.%N)
whisper_duration=$(echo "$end_time - $start_time" | bc)

# 2. Test Parakeet
echo "Sending to Parakeet-TDT..."
start_time=$(date +%s.%N)
parakeet_res=$(curl -sS -X POST "$PARAKEET_URL" \
    -H "Accept: application/json" \
    -F "file=@${AUDIO_FILE}" \
    -F "prompt=${PROMPT}" \
    -F "response_format=json")
end_time=$(date +%s.%N)
parakeet_duration=$(echo "$end_time - $start_time" | bc)

# Output Results
echo ""
echo "==== RESULTS ===="
echo "--- speaches.ai (Whisper) ---"
echo "Time: ${whisper_duration}s"
echo "Text: $(echo $whisper_res | jq -r '.text')"
echo ""
echo "--- Parakeet-TDT ---"
echo "Time: ${parakeet_duration}s"
echo "Text: $(echo $parakeet_res | jq -r '.text')"
echo "--------------------------------------------------------"

# Hallucination Check
if [[ $(echo $whisper_res | jq -r '.text') == *"$PROMPT"* ]]; then
    echo "ALERT: Whisper hallucination detected (prompt-bleeding)."
fi