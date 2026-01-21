import json
import os
import torch
import sys
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

# Configuration
MODEL_ID = "google/gemma-3-270m-it"
CHUNK_SIZE = 1000  # Characters for rough chunking

# --- Pydantic Schema Definition (Step 3) ---

class Transition(BaseModel):
    from_state: str = Field(..., alias="from")
    to_state: str = Field(..., alias="to")
    condition: str = Field(..., description="Condition string e.g. event == 'alert'")

class StateAction(BaseModel):
    description: str
    actions: List[str]

class StateMachine(BaseModel):
    initial_state: str
    states: Dict[str, StateAction]
    transitions: List[Transition]

class Overlay(BaseModel):
    mission_id: str
    state_machine: StateMachine

# --- Helpers ---

def read_pdf(file_path):
    print(f"Reading PDF: {file_path}")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return ""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def chunk_text(text, size=CHUNK_SIZE):
    return [text[i:i+size] for i in range(0, len(text), size)]

def load_model():
    print(f"Initializing Gemma 3 ({MODEL_ID})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map="auto", 
            dtype=torch.float32  # Force FP32 for stability with small model
        )
        return tokenizer, model
    except OSError as e:
        if "gated repo" in str(e) or "401" in str(e):
            print("\nCRITICAL ERROR: Access Denied to Gated Model.")
            print(f"Please ensure you have access to {MODEL_ID} on Hugging Face.")
            print("Then run: huggingface-cli login")
            print("Or set HF_TOKEN environment variable.\n")
        raise e

def generate_text(model, tokenizer, prompt, max_new_tokens=1024):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the NEW tokens
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"DEBUG Response: {response}") # Optional debug
    return response

def extract_json_from_response(response_text):
    # print(f"\n--- RAW MODEL OUTPUT START ---\n{response_text}\n--- RAW MODEL OUTPUT END ---\n")
    try:
        # Robust extraction: find the first outer { and the last outer }
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            return json.loads(json_str)
        else:
            print("No JSON object found (missing braces).")
            return None
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return None

# --- Core Logic ---

def extract_security_info(chunk, model, tokenizer):
    prompt = f"""<start_of_turn>user
Analyze the following text chunk. Extract ANY information related to:
- Security protocols
- Threat models or threat levels
- Operational constraints or domain rules
- Identifying unauthorized actors

Chunk:
{chunk}

Output a concise summary list. If nothing relevant, say "None".
<end_of_turn>
<start_of_turn>model
"""
    return generate_text(model, tokenizer, prompt, max_new_tokens=256)

def generate_canonical_context(mission_context, aggregated_info, model, tokenizer):
    print("Generating Canonical Context...")
    prompt = f"""<start_of_turn>user
You are a high-level security mission analyst. 
Your task is to ENHANCE the provided Mission Context using the Aggregated Security Info.

INPUT 1: Mission Context
{json.dumps(mission_context, indent=2)}

INPUT 2: Aggregated Security Info
{aggregated_info}

INSTRUCTIONS:
Create a NEW JSON object called 'Canonical Mission Context'.
It MUST contain:
- "original_mission_id": from Mission Context locaton.
- "derived_security_profile": Object with environment and threat_model based on Security Info.
- "operational_constraints": List of constraints from Security Info.
- "implicit_context_expansion": A summary string.

Do not just copy the input. Synthesize a new JSON.

Output JSON ONLY.
<end_of_turn>
<start_of_turn>model
```json
"""
    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    return extract_json_from_response(response)

def generate_overlay(canonical_context, model, tokenizer):
    print("Generating Overlay with Pydantic Schema...")
    # schema_json = Overlay.model_json_schema() # Schema is too large for 270M prompt usually, let's simplify prompt
    
    prompt = f"""<start_of_turn>user
Task: Generate a State Machine Overlay JSON based on the Canonical Mission Context.

INPUT Context:
{json.dumps(canonical_context, indent=2)}

INSTRUCTIONS:
Generate a JSON object with the following structure:
{{
  "mission_id": "string",
  "state_machine": {{
    "initial_state": "Patrol",
    "states": {{
      "Patrol": {{ "description": "...", "actions": ["..."] }},
      "Investigate": {{ "description": "...", "actions": ["..."] }},
      "Alert": {{ "description": "...", "actions": ["..."] }}
    }},
    "transitions": [
      {{ "from": "Patrol", "to": "Investigate", "condition": "event == 'suspicious'" }},
      {{ "from": "Investigate", "to": "Alert", "condition": "event == 'confirmed'" }}
    ]
  }}
}}

Ensure valid JSON.
<end_of_turn>
<start_of_turn>model
```json
"""
    response = generate_text(model, tokenizer, prompt, max_new_tokens=2048)
    data = extract_json_from_response(response)
    
    if data:
        try:
            print("Validating with Pydantic...")
            overlay = Overlay.model_validate(data)
            return overlay.model_dump(by_alias=True)
        except ValidationError as e:
            print(f"Pydantic Validation Error: {e}")
            # print(f"Data: {json.dumps(data, indent=2)}")
            return None
    return None

def main():
    print("Starting Combined Mission Generator (Step 2 & 3)...")
    
    # 0. Load Model ONCE
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 1. Load Mission Context
    try:
        with open('MissionContext.json', 'r') as f:
            mission_context = json.load(f)
    except FileNotFoundError:
        print("MissionContext.json not found.")
        sys.exit(1)

    # 2. Read PDF Documents (or txt fallback)
    pdf_path = 'data/documents.pdf'
    txt_path = 'data/documents.txt'
    
    doc_text = ""
    if os.path.exists(pdf_path):
        doc_text = read_pdf(pdf_path)
    elif os.path.exists(txt_path):
        print("PDF not found, falling back to text file.")
        with open(txt_path, 'r') as f:
            doc_text = f.read()
    else:
        print("No document found in data/")
        sys.exit(1)

    # 3. Process Documents (Chunk & Extract)
    print("Processing Document Chunks...")
    chunks = chunk_text(doc_text)
    aggregated_info = ""
    for i, chunk in enumerate(chunks[:5]): # Limit to first 5 chunks for demo efficiency
        print(f" - Analyzing chunk {i+1}/{len(chunks)}...")
        info = extract_security_info(chunk, model, tokenizer)
        if "None" not in info:
            aggregated_info += f"\nChunk {i+1} summary: {info}"

    # 4. Generate Canonical Context
    canonical_context = generate_canonical_context(mission_context, aggregated_info, model, tokenizer)
    if canonical_context:
        with open('CanonicalMissionContext.json', 'w') as f:
            json.dump(canonical_context, f, indent=2)
        print("Success: Created CanonicalMissionContext.json")
    else:
        print("Failed to generate Canonical Context")
        sys.exit(1)

    # 5. Generate Overlay
    overlay = generate_overlay(canonical_context, model, tokenizer)
    if overlay:
        with open('OverLay.json', 'w') as f:
            json.dump(overlay, f, indent=2)
        print("Success: Created OverLay.json")
    else:
        print("Failed to generate Overlay")
        sys.exit(1)

if __name__ == "__main__":
    main()
