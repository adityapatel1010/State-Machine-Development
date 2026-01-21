#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "==========================================="
echo "Running Step 1: Mission Context..."
echo "==========================================="
python3 step1_mission_context.py

echo ""
echo "==========================================="
echo "Running Step 2: Canonical Context (Gemma 3)..."
echo "==========================================="
python3 step2_canonical_context.py

echo ""
echo "==========================================="
echo "Running Step 3: Overlay Generation (Gemma 3)..."
echo "==========================================="
python3 step3_overlay.py

echo ""
echo "==========================================="
echo "Running Step 4: Compiler..."
echo "==========================================="
python3 step4_compiler.py

echo ""
echo "==========================================="
echo "Running Step 5: Runtime Simulation..."
echo "==========================================="
python3 step5_runtime.py

echo ""
echo "Pipeline completed successfully."
