# State Machine Generator Pipeline

This project generates a State Machine DSL from high-level mission requirements using a 5-step pipeline integrated with Gemma 3.

## Prerequisites

- Python 3.8+
- A machine with a GPU is recommended for Gemma 3 inference (steps 2 & 3), but it can run on CPU (slowly).
- Hugging Face Access Token (for Gemma 3 models). You may need to log in via `huggingface-cli login`.

## Installation

1.  **Clone/Navigate** to this directory.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the pipeline steps sequentially:

### Step 1: Mission Context
Merges your input (`input_mission.json`) with the security template.
```bash
python3 step1_mission_context.py
```

### Step 2: Canonical Context (Gemma 3)
Uses LLM to analyze the context and documents.
```bash
python3 step2_canonical_context.py
```

### Step 3: Overlay Generation (Gemma 3)
Uses LLM to generate the state machine logic with Pydantic validation.
```bash
python3 step3_overlay.py
```

### Step 4: Compiler
Compiles and validates the overlay into the final DSL.
```bash
python3 step4_compiler.py
```

### Step 5: Runtime Simulation
Simulates the execution of the generated state machine.
```bash
python3 step5_runtime.py
```

## Configuration

- **Input**: Modify `input_mission.json` to change the mission parameters.
- **Templates**: Modify `templates/generic_template.json` for base security settings.
- **Documents**: Update `data/documents.txt` to change the knowledge base.
