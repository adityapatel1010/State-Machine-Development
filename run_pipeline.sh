#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

echo "==========================================="
echo "Running Step 1: Mission Context..."
echo "==========================================="
python3 step1_mission_context.py

echo ""
echo "==========================================="
echo "Running Combined Step 2 & 3: Mission Generator (Gemma 3)..."
echo "==========================================="
python3 step2_3_mission_generator.py

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
