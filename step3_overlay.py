import json
import os
import torch
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_ID = "google/gemma-3-270m-it"

# --- Pydantic Schema Definition ---

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

# --- Inference Logic ---

def gemma_3_generate_overlay(canonical_context):
    print(f"Initializing Gemma 3 ({MODEL_ID})...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map="auto", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Get Schema JSON for prompt
    schema_json = Overlay.model_json_schema()
    
    prompt = f"""<start_of_turn>user
You are a mission planning system.
Based on the provided Canonical Mission Context, generate a State Machine Overlay.

## Canonical Mission Context
{json.dumps(canonical_context, indent=2)}

## Output Requirement
You MUST output a valid JSON object that strictly adheres to the following JSON Schema:
{json.dumps(schema_json, indent=2)}

Ensure the 'initial_state' exists in the 'states' dictionary.
Ensure all transitions reference valid states.
Output ONLY the JSON object.
<end_of_turn>
<start_of_turn>model
```json
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=2048,
        temperature=0.2,
        do_sample=True
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON
    try:
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            
        print("Parsing and Validating JSON with Pydantic...")
        # Validate with Pydantic
        overlay_data = Overlay.model_validate_json(json_str)
        return overlay_data.model_dump(by_alias=True) # Dump back to dict (using aliases for 'from'/'to')
        
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e}")
        print(f"Raw JSON attempted: {json_str[:500]}...")
        return None
    except Exception as e:
        print(f"Error parsing JSON output: {e}")
        # print(f"Raw Output: {response_text}")
        return None

def main():
    print("Step 3: Generating OverLay.json (Gemma 3 + Pydantic)...")
    
    # 1. Load CanonicalMissionContext.json
    try:
        with open('CanonicalMissionContext.json', 'r') as f:
            canonical_context = json.load(f)
            print("Loaded CanonicalMissionContext.json")
    except FileNotFoundError:
        print("Error: CanonicalMissionContext.json not found. Run Step 2 first.")
        return

    # 2. Gemma 3 Inference
    overlay = gemma_3_generate_overlay(canonical_context)
    
    if overlay:
        # 3. Save Output
        with open('OverLay.json', 'w') as f:
            json.dump(overlay, f, indent=2)
        print("Success: Created OverLay.json")
    else:
        print("Failed to generate valid Overlay.")

if __name__ == "__main__":
    main()
