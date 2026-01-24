import json
import os
import torch
import sys
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader

# Configuration
MODEL_ID = "google/gemma-3-270m-it"
CHUNK_SIZE = 1000  # Characters for rough chunking

# Sample States for prompt customization
SAMPLE_STATES = """
{
  "mission_id": "lunar_rover_mission_001",
  "initial_state": "powered_off",
  "states": {
    "powered_off": {
      "description": "Rover is powered down prior to mission start.",
      "actions": [
        "monitor_battery_health",
        "await_start_command"
      ]
    },
    "booting": {
      "description": "Rover systems are booting and performing initial checks.",
      "actions": [
        "initialize_computers",
        "run_self_diagnostics",
        "calibrate_sensors"
      ]
    },
    "idle": {
      "description": "Rover is ready but awaiting further instructions.",
      "actions": [
        "maintain_thermal_balance",
        "listen_for_commands",
        "low_power_monitoring"
      ]
    },
    "navigating": {
      "description": "Rover is traversing the lunar surface to a target location.",
      "actions": [
        "plan_path",
        "avoid_obstacles",
        "drive_motors",
        "update_position_estimate"
      ]
    },
    "science_operations": {
      "description": "Rover is conducting scientific measurements and experiments.",
      "actions": [
        "deploy_instruments",
        "collect_samples",
        "analyze_soil",
        "store_science_data"
      ]
    },
    "communicating": {
      "description": "Rover is transmitting or receiving data from mission control.",
      "actions": [
        "establish_link",
        "transmit_telemetry",
        "receive_commands"
      ]
    },
    "charging": {
      "description": "Rover is replenishing power using solar panels.",
      "actions": [
        "orient_to_sun",
        "deploy_solar_panels",
        "charge_batteries"
      ]
    },
    "safe_mode": {
      "description": "Rover has detected an anomaly and entered a protective state.",
      "actions": [
        "shutdown_non_essential_systems",
        "log_fault_data",
        "attempt_self_recovery"
      ]
    },
    "mission_complete": {
      "description": "Mission objectives have been completed.",
      "actions": [
        "final_data_transmission",
        "power_down_non_critical_systems"
      ]
    }
  },
  "transitions": [
    {
      "from": "powered_off",
      "to": "booting",
      "condition": "event == 'start_mission'"
    },
    {
      "from": "booting",
      "to": "idle",
      "condition": "diagnostics_passed == true"
    },
    {
      "from": "idle",
      "to": "navigating",
      "condition": "event == 'navigate_to_target'"
    },
    {
      "from": "navigating",
      "to": "science_operations",
      "condition": "at_target_location == true"
    },
    {
      "from": "science_operations",
      "to": "communicating",
      "condition": "data_ready == true"
    },
    {
      "from": "communicating",
      "to": "idle",
      "condition": "communication_complete == true"
    },
    {
      "from": "idle",
      "to": "charging",
      "condition": "battery_level < 20"
    },
    {
      "from": "charging",
      "to": "idle",
      "condition": "battery_level >= 80"
    },
    {
      "from": "idle",
      "to": "safe_mode",
      "condition": "event == 'critical_fault'"
    },
    {
      "from": "safe_mode",
      "to": "idle",
      "condition": "fault_resolved == true"
    },
    {
      "from": "idle",
      "to": "mission_complete",
      "condition": "event == 'end_mission'"
    }
  ]
}
"""

# --- Helpers ---

def read_pdf(file_path):
    """Read text from PDF file"""
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
    """Split text into chunks for processing"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def load_model():
    """Load the Gemma model and tokenizer"""
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
    """Generate text from model given a prompt"""
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
    print(f"DEBUG Response: {response}")  # Show first 200 chars
    return response

def extract_json_from_response(response_text):
    """Extract JSON object from model response"""
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
        print(f"Attempted to parse: {response_text[start:end][:500]}...")
        return None

# --- Validation Functions ---

def validate_canonical_context(data):
    """Simple validation for canonical context"""
    required = ["mission_id", "key_variables", "key_states", "constraints"]
    if not all(k in data for k in required):
        return False, f"Missing required keys. Need: {required}, Got: {list(data.keys())}"
    if not isinstance(data["key_variables"], list):
        return False, "key_variables must be a list"
    if not isinstance(data["key_states"], list):
        return False, "key_states must be a list"
    if not isinstance(data["constraints"], list):
        return False, "constraints must be a list"
    return True, "Valid"

def validate_overlay(data):
    """Simple validation for overlay"""
    required = ["mission_id", "initial_state", "states", "transitions"]
    if not all(k in data for k in required):
        return False, f"Missing required keys. Need: {required}, Got: {list(data.keys())}"
    if not isinstance(data["states"], dict):
        return False, "states must be an object/dict"
    if not isinstance(data["transitions"], list):
        return False, "transitions must be a list"
    
    # Validate state structure
    for state_name, state_data in data["states"].items():
        if "description" not in state_data or "actions" not in state_data:
            return False, f"State '{state_name}' missing description or actions"
        if not isinstance(state_data["actions"], list):
            return False, f"State '{state_name}' actions must be a list"
    
    # Validate transition structure
    for i, trans in enumerate(data["transitions"]):
        if not all(k in trans for k in ["from", "to", "condition"]):
            return False, f"Transition {i} missing required fields (from, to, condition)"
    
    return True, "Valid"

# --- Core Logic ---

def extract_security_info(chunk, model, tokenizer):
    """Extract security and operational information from a text chunk"""
    prompt = f"""<start_of_turn>user
Analyze the following text chunk. Extract ANY information related to:
- Security protocols
- Threat models or threat levels
- Operational constraints or domain rules
- Variables that need to be tracked (like battery, position, status)
- System states or modes
- Identifying unauthorized actors

Chunk:
{chunk}

Output a concise summary list. If nothing relevant, say "None".
<end_of_turn>
<start_of_turn>model
"""
    return generate_text(model, tokenizer, prompt, max_new_tokens=256)

def generate_canonical_context(mission_context, aggregated_info, model, tokenizer):
    """Generate canonical context identifying key variables and states"""
    print("Generating Canonical Context...")
    prompt = f"""<start_of_turn>user
You are analyzing a mission to identify key states and variables.

MISSION CONTEXT:
{json.dumps(mission_context, indent=2)}

SECURITY/OPERATIONAL INFO:
{aggregated_info}

REFERENCE EXAMPLE (for understanding state machines):
{SAMPLE_STATES}

YOUR TASK:
1. Identify KEY VARIABLES that will change during the mission (e.g., battery_level, position, connection_status)
2. Identify KEY STATES the system can be in (e.g., idle, active, charging, emergency)
3. List CONSTRAINTS or rules from the security info

OUTPUT REQUIREMENTS:
- Output ONLY valid JSON
- Use this exact structure:
{{
  "mission_id": "<mission_id_from_context>",
  "key_variables": [
    {{"name": "variable_name", "type": "numeric|boolean|string", "description": "what it tracks"}}
  ],
  "key_states": [
    {{"name": "state_name", "description": "what this state means"}}
  ],
  "constraints": ["constraint1", "constraint2"]
}}

Generate the JSON now:
<end_of_turn>
<start_of_turn>model
{{
"""
    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    
    # Add opening brace if not present
    if not response.strip().startswith('{'):
        response = '{' + response
    
    data = extract_json_from_response(response)
    
    # Validate
    if data:
        is_valid, msg = validate_canonical_context(data)
        if is_valid:
            print(f"✓ Canonical Context validated: {msg}")
            return data
        else:
            print(f"✗ Validation failed: {msg}")
            return None
    return None

def generate_overlay(canonical_context, model, tokenizer):
    """Generate state machine overlay from canonical context"""
    print("Generating Overlay...")
    
    prompt = f"""<start_of_turn>user
You are creating a state machine for a mission.

INPUT (Variables and States identified):
{json.dumps(canonical_context, indent=2)}

YOUR TASK:
Create a complete state machine with:
1. All states from key_states, with specific actions for each
2. Transitions between states based on the key_variables
3. Conditions using the variables (e.g., "battery_level < 20", "status == 'ready'")

OUTPUT REQUIREMENTS:
- Output ONLY valid JSON
- mission_id MUST match the input: "{canonical_context.get('mission_id', '')}"
- Each state needs a description and list of actions
- Each transition needs: from, to, and condition
- Use this EXACT structure:

{{
  "mission_id": "{canonical_context.get('mission_id', '')}",
  "initial_state": "<starting_state>",
  "states": {{
    "state_name": {{
      "description": "description here",
      "actions": ["action1", "action2"]
    }}
  }},
  "transitions": [
    {{
      "from": "state1",
      "to": "state2",
      "condition": "variable_name > value"
    }}
  ]
}}

Generate the complete state machine JSON now:
<end_of_turn>
<start_of_turn>model
{{
"""
    response = generate_text(model, tokenizer, prompt, max_new_tokens=2048)
    
    # Add opening brace if not present
    if not response.strip().startswith('{'):
        response = '{' + response
    
    data = extract_json_from_response(response)
    
    # Validate
    if data:
        is_valid, msg = validate_overlay(data)
        if is_valid:
            print(f"✓ Overlay validated: {msg}")
            return data
        else:
            print(f"✗ Validation failed: {msg}")
            return None
    return None

def main():
    """Main execution function"""
    print("=" * 60)
    print("Starting Combined Mission Generator")
    print("=" * 60)
    
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
        print(f"✓ Loaded MissionContext.json")
    except FileNotFoundError:
        print("✗ MissionContext.json not found.")
        sys.exit(1)

    # 2. Read PDF Documents (or txt fallback)
    pdf_path = 'data/documents.pdf'
    txt_path = 'data/documents.txt'
    
    doc_text = ""
    if os.path.exists(pdf_path):
        doc_text = read_pdf(pdf_path)
        print(f"✓ Loaded PDF document ({len(doc_text)} characters)")
    elif os.path.exists(txt_path):
        print("PDF not found, falling back to text file.")
        with open(txt_path, 'r') as f:
            doc_text = f.read()
        print(f"✓ Loaded text document ({len(doc_text)} characters)")
    else:
        print("✗ No document found in data/")
        sys.exit(1)

    # 3. Process Documents (Chunk & Extract)
    print("\n" + "=" * 60)
    print("Processing Document Chunks...")
    print("=" * 60)
    chunks = chunk_text(doc_text)
    aggregated_info = ""
    
    # Limit to first 5 chunks for demo efficiency
    max_chunks = min(5, len(chunks))
    for i, chunk in enumerate(chunks[:max_chunks]):
        print(f" - Analyzing chunk {i+1}/{max_chunks}...")
        info = extract_security_info(chunk, model, tokenizer)
        if "None" not in info and info.strip():
            aggregated_info += f"\nChunk {i+1} summary: {info}"
    
    if not aggregated_info.strip():
        aggregated_info = "No specific security or operational constraints found in documents."
    
    print(f"✓ Aggregated info from {max_chunks} chunks")

    # 4. Generate Canonical Context
    print("\n" + "=" * 60)
    print("Step 1: Generating Canonical Context")
    print("=" * 60)
    canonical_context = generate_canonical_context(mission_context, aggregated_info, model, tokenizer)
    
    if canonical_context:
        output_file = 'CanonicalMissionContext.json'
        with open(output_file, 'w') as f:
            json.dump(canonical_context, f, indent=2)
        print(f"✓ Success: Created {output_file}")
        print(f"  - Mission ID: {canonical_context.get('mission_id')}")
        print(f"  - Variables: {len(canonical_context.get('key_variables', []))}")
        print(f"  - States: {len(canonical_context.get('key_states', []))}")
    else:
        print("✗ Failed to generate Canonical Context")
        sys.exit(1)

    # 5. Generate Overlay
    print("\n" + "=" * 60)
    print("Step 2: Generating State Machine Overlay")
    print("=" * 60)
    overlay = generate_overlay(canonical_context, model, tokenizer)
    
    if overlay:
        output_file = 'OverLay.json'
        with open(output_file, 'w') as f:
            json.dump(overlay, f, indent=2)
        print(f"✓ Success: Created {output_file}")
        print(f"  - Mission ID: {overlay.get('mission_id')}")
        print(f"  - Initial State: {overlay.get('initial_state')}")
        print(f"  - Total States: {len(overlay.get('states', {}))}")
        print(f"  - Total Transitions: {len(overlay.get('transitions', []))}")
    else:
        print("✗ Failed to generate Overlay")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tasks completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()