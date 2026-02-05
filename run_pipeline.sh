#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================================"
echo "Starting Mission Generation Pipeline"
echo "============================================================"

# Check for dependencies
# You might want to uncomment this if you need to ensure dependencies are installed every time
# pip install -r requirements.txt

# Step 1: Mission Context
echo ""
echo "------------------------------------------------------------"
echo "[Step 1] Generating Mission Context..."
echo "------------------------------------------------------------"
if [ ! -f "input_mission.json" ]; then
    echo "Warning: input_mission.json not found in current directory."
    if [ -f "previous_implementation/input_mission.json" ]; then
        echo "Found in previous_implementation, copying..."
        cp previous_implementation/input_mission.json .
    else
        echo "Error: input_mission.json missing."
        exit 1
    fi
fi

python3 step1_mission_context.py

# Step 2: State Generation
echo ""
echo "------------------------------------------------------------"
echo "[Step 2] Generating Mission States (RAG + VLM Analysis)..."
echo "------------------------------------------------------------"

# Check for HF Token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN environment variable is not set."
    echo "If you have not logged in via 'huggingface-cli login', this step may fail."
fi

python3 step2_state_generation.py

echo ""
echo "============================================================"
echo "Pipeline Completed Successfully!"
echo "============================================================"
