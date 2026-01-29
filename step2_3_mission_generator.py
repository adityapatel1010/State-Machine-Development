import json
import os
import torch
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import docx
import openpyxl

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
CHUNK_SIZE = 1000  # Characters for rough chunking

# Sample States for prompt customization - Industrial Anomaly Detection
SAMPLE_STATES = """
- Normal: System operating within all safety parameters.
- Monitoring: Minor deviation detected; increased sampling rate active.
- Escalation: Thresholds exceeded; automated mitigation (e.g. pumps) engaged.
- Alert: Critical hazard condition; full evacuation and shutdown.
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
            text_extracted = page.extract_text()
            if text_extracted:
                text += text_extracted + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def read_docx(file_path):
    """Read text from DOCX file"""
    print(f"Reading DOCX: {file_path}")
    try:
        doc = docx.Document(file_path)
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading DOCX: {e}")
        return ""

def read_excel(file_path):
    """Read text from Excel file (all sheets)"""
    print(f"Reading Excel: {file_path}")
    try:
        dfs = pd.read_excel(file_path, sheet_name=None)
        text = []
        for sheet_name, df in dfs.items():
            text.append(f"--- Sheet: {sheet_name} ---")
            text.append(df.to_string())
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return ""

def read_document(file_path):
    """Generic document reader dispatcher"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return read_docx(file_path)
    elif ext in ['.xlsx', '.xls']:
        return read_excel(file_path)
    elif ext in ['.txt', '.md', '.json', '.log']:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file: {e}")
            return ""
    else:
        print(f"Unsupported file format: {ext}")
        return ""

def get_relevant_chunks(chunks, query, top_k=5):
    """RAG: Select top_k chunks most relevant to the query"""
    print("Initializing Sentence Transformer for RAG...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(chunks)} chunks...")
    corpus_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Cosine similarity
    from sentence_transformers import util
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Get top k
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    
    relevant_chunks = []
    print("\nTop Relevant Chunks:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"  - Score: {score:.4f}")
        relevant_chunks.append(chunks[idx])
        
    return relevant_chunks

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
        temperature=0.1,  # Slight temperature for better diversity
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        top_p=0.9
    )
    
    # Decode only the NEW tokens
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Show response length info
    # print(f"Generated {len(generated_tokens)} tokens (~{len(response)} chars)")
    if len(response) > 200:
        print(f"Preview: {response}")
    else:
        print(f"Full response: {response}")
    
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
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        # Try to fix common issues
        start = response_text.find("{")
        if start == -1:
            print("No opening brace found")
            return None
            
        # Count braces to find where JSON likely ends
        brace_count = 0
        end = start
        for i in range(start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        if end > start:
            json_str = response_text[start:end]
            print(f"Attempting to parse with brace counting (length: {len(json_str)})...")
            try:
                return json.loads(json_str)
            except:
                print(f"Still failed. Response preview: {json_str[:300]}...")
                return None
        else:
            print(f"Could not find matching braces. Response: {response_text[:500]}...")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# --- Validation Functions ---

def validate_canonical_context(data):
    """Simple validation for canonical context"""
    required = ["mission_id", "key_variables", "constraints"]
    if not all(k in data for k in required):
        return False, f"Missing required keys. Need: {required}, Got: {list(data.keys())}"
    if not isinstance(data["key_variables"], list):
        return False, "key_variables must be a list"
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
        if "description" not in state_data:
            return False, f"State '{state_name}' missing description"
        # if not isinstance(state_data["actions"], list):
        #     return False, f"State '{state_name}' actions must be a list"
    
    # Validate transition structure
    # Validate transition structure
    for i, trans in enumerate(data["transitions"]):
        if not all(k in trans for k in ["from", "to"]):
            return False, f"Transition {i} missing required fields (from, to)"
            
        # condition is optional initially (added by condition_generator)
    
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
    return generate_text(model, tokenizer, prompt, max_new_tokens=300)

def generate_canonical_context(mission_context, aggregated_info, model, tokenizer):
    """Generate canonical context identifying key variables and states"""
    print("Generating Canonical Context...")
    prompt = f"""<start_of_turn>user
You are analyzing a mission to identify UNIQUE states and variables specific to this domain.

MISSION CONTEXT:
{json.dumps(mission_context, indent=2)}

SECURITY/OPERATIONAL INFO (Domain Knowledge):
{aggregated_info}

YOUR TASK:
Identify KEY VARIABLES and POTENTIAL STATES that are unique to this domain.
Do not output generic placeholders. Use the domain terms found in the documents.

OUTPUT REQUIREMENTS:
- Output ONLY valid JSON.
- "key_variables": 5-8 variables critical for this specific mission.
- "constraints": Operational limits found in the documents.
- "unique_states": A list of potential states implied by the documents.

Use EXACTLY this structure:
{{
  "mission_id": "<use the exact mission_id from context>",
  "key_variables": [
    {{"name": "variable_name", "type": "numeric/string", "description": "Description"}}
  ],
  "constraints": ["Constraint 1", "Constraint 2"],
  "unique_states": ["State1", "State2"]
}}

IMPORTANT: Output ONLY the JSON object above.
<end_of_turn>
<start_of_turn>model
{{
"""
    # Increase token limit to ensure complete response
    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    
    # Add opening brace if not present
    if not response.strip().startswith('{'):
        response = '{' + response
    
    data = extract_json_from_response(response)
    
    # Validate - update validation to not require key_states
    if data:
        # Remove key_states if the model included it anyway
        if "key_states" in data:
            print("⚠ Model included key_states, removing it...")
            del data["key_states"]
        
        required = ["mission_id", "key_variables", "constraints"]
        if not all(k in data for k in required):
            print(f"✗ Missing required keys. Need: {required}, Got: {list(data.keys())}")
            return None
        if not isinstance(data["key_variables"], list):
            print("✗ key_variables must be a list")
            return None
        if not isinstance(data["constraints"], list):
            print("✗ constraints must be a list")
            return None
        print(f"✓ Canonical Context validated")
        return data
    return None

def generate_overlay(canonical_context, model, tokenizer):
    """Generate state machine overlay from canonical context"""
    print("Generating Overlay...")
    
    # Extract variable names for easier reference
    variable_names = [v["name"] for v in canonical_context.get("key_variables", [])]

    prompt = f"""<start_of_turn>user
You are StateMachineProtocolCompiler.

SOURCE OF TRUTH
MISSION CONTEXT:
{json.dumps(canonical_context, indent=2)}

HARD REQUIREMENTS
1) EXACTLY 10 states total.
2) Must include core states exactly: Normal, Escalation, Alert, Inform.
3) The remaining 6 states MUST be sensor/VLM-evidence-driven operational modes implied by mission context.
6) Every state must be reachable from Normal (possibly via other states).
7) No dead states: each state must have at least one inbound and one outbound transition (except Alert may be terminal ONLY if mission context says so).
8) Protocol logic required:
   - Any "Assessment" type state MUST branch to:
     (a) mitigation or escalation if hazard confirmed/persistent
     (b) return to Normal if cleared
   - Alert must only be entered from Escalation or from a confirmed severe hazard state (not directly from Normal unless mission context explicitly allows).

TRANSITIONS (MANDATORY)
For each non-core custom state you create, you MUST define:
- a "confirmed" exit transition (hazard confirmed/persists -> mitigation/escalation)
- a "cleared" exit transition (evidence clears -> Normal or lower severity)
Do NOT define conditions for transitions.

OUTPUT FORMAT
Output ONLY valid JSON with EXACT structure:

{{
  "mission_id": "{canonical_context.get('mission_id','')}",
  "initial_state": "Normal",
  "states": {{
    "Normal": {{"description": "..."}},
    "Escalation": {{"description": "..."}},
    "Alert": {{"description": "..."}},
    "Inform": {{"description": "..."}}
    // plus exactly 6 custom states
  }},
  "transitions": [
    {{"from": "...", "to": "..."}}
  ]
}}

FINAL SILENT VALIDATION (do not output)
- 10 states exactly
- all states reachable
- assessment states have confirmed + cleared exits
- Alert only via Escalation or confirmed severe hazard state
<end_of_turn>
<start_of_turn>model
{{"""

#     prompt = f"""<start_of_turn>user
# You are StateMachineCompiler.

# SOURCE OF TRUTH
# MISSION CONTEXT:
# {json.dumps(canonical_context, indent=2)}

# ALLOWED VARIABLES (closed world)
# You may ONLY reference these variables in transition conditions.
# You may NOT invent new variables, aliases, or helper flags.
# Allowed: {', '.join(variable_names)}

# REFERENCE (STYLE ONLY, NOT CONTENT)
# {SAMPLE_STATES}
# Do NOT copy state names or thresholds from reference.

# HARD REQUIREMENT: EXACTLY 10 STATES TOTAL
# You MUST output exactly 10 state objects in "states":
# - 4 fixed core states (must exist exactly once, exact names):
#   1) Normal
#   2) Escalation
#   3) Alert
#   4) Inform
# - PLUS exactly 6 custom states derived from the mission context.

# CUSTOM STATE DERIVATION (NO GUESSWORK)
# Derive the 6 custom states by extracting the 6 most important "required operational modes" implied by the mission context.
# A "required operational mode" must be justified by explicit fields in the mission context (e.g., mission goals, constraints, sensors/feeds, policies, safety requirements, comms constraints, resource constraints, operator workflow).
# Do NOT invent modes that are not supported by mission context.

# CUSTOM STATE NAMING
# - PascalCase, 1-3 words
# - Not generic (avoid Monitoring, Handling, Managing, Processing unless mission context explicitly calls for it)
# - Must be distinct and cover different mission needs (no near-duplicates)

# ACTIONS
# Each state MUST have exactly 3 actions.
# Actions MUST be snake_case and must be feasible given mission context (no invented integrations).
# Actions should be concrete and operational.

# TRANSITIONS (STRICT)
# - Provide 12-20 transitions.
# - Every transition condition MUST use ONLY ALLOWED VARIABLES (verbatim).
# - No unconditional conditions like "true" or "always".
# - No made-up thresholds. Thresholds may only appear if the mission context explicitly contains them.
#   If context does NOT contain numeric thresholds, use relative comparisons or discrete enums already present in ALLOWED VARIABLES.
# - Ensure:
#   (a) At least 1 transition OUT of every state.
#   (b) At least 1 transition INTO every custom state.
#   (c) At least 1 de-escalation path back to Normal from Escalation/Alert/Inform.
#   (d) Alert must be reachable from Escalation (directly or indirectly).

# INITIAL STATE
# initial_state MUST be "Normal"

# OUTPUT FORMAT
# Output ONLY valid JSON and EXACTLY this structure:

# {{
#   "mission_id": "{canonical_context.get('mission_id','')}",
#   "initial_state": "Normal",
#   "states": {{
#     "Normal": {{"description": "...", "actions": ["...", "...", "..."]}},
#     "Escalation": {{"description": "...", "actions": ["...", "...", "..."]}},
#     "Alert": {{"description": "...", "actions": ["...", "...", "..."]}},
#     "Inform": {{"description": "...", "actions": ["...", "...", "..."]}}
#     // plus exactly 6 custom states
#   }},
#   "transitions": [
#     {{"from": "...", "to": "...", "condition": "..."}}
#   ]
# }}

# FINAL SILENT VALIDATION (do not output)
# - Exactly 10 states total
# - Exactly 6 custom states
# - Exactly 3 actions per state
# - Conditions reference ONLY ALLOWED VARIABLES
# - No "true" condition
# - No numeric thresholds unless present in mission context
# <end_of_turn>
# <start_of_turn>model
# {{"""

#     prompt = f"""<start_of_turn>user
# You are StateMachineCompiler.

# GOAL
# Generate a mission state machine that is:
# - grounded ONLY in the provided mission context + allowed variables
# - consistent, minimal, and reliable
# - NOT a copy of the reference states (reference is style inspiration only)

# MISSION CONTEXT (SOURCE OF TRUTH)
# {json.dumps(canonical_context, indent=2)}

# ALLOWED VARIABLES (closed world)
# You may ONLY reference these variables in transition conditions (no new variables, no synonyms):
# {', '.join(variable_names)}

# REFERENCE STATE PATTERNS (INSPIRATION ONLY)
# These are examples of wording/action style. Do NOT reuse state names or transitions unless they are truly required by THIS mission context.
# {SAMPLE_STATES}

# MANDATORY CORE STATES
# You MUST include exactly these 4 core states by name (these are always present):
# - Normal
# - Escalation
# - Alert
# - Inform

# CUSTOM STATES
# You MAY add 1-4 additional mission-specific states if the mission context clearly demands them.
# Rules for custom state naming:
# - PascalCase
# - 1-3 words max
# - describe an observable operational mode (not a vague concept)

# ACTIONS RULES (anti-hallucination)
# Each state must have 2-3 actions.
# Actions must be simple verbs + object (snake_case), and must be feasible from the context.
# Examples of style: "monitor_feeds", "log_event", "notify_operator"
# Do NOT invent external systems, integrations, or sensors not implied by the mission context.

# TRANSITION RULES (anti-guesswork)
# - Every transition condition MUST use ONLY ALLOWED VARIABLES.
# - If you cannot express a transition condition using ONLY allowed variables, DO NOT include that transition.
# - Prefer simple conditions (comparisons, boolean checks) over complex logic.
# - No probabilities, no made-up thresholds unless the context provides them.
# - Ensure reachability: at least one transition out of each non-terminal state.
# - Ensure closure: at least one path back to Normal from Escalation/Alert/Inform.

# OUTPUT SIZE
# - Total states: 5-8 (including the 4 mandatory core states)
# - Transitions: enough to make the machine coherent (typically 6-14)

# OUTPUT FORMAT (STRICT)
# Output ONLY valid JSON (no markdown, no comments, no extra text).
# The JSON MUST match EXACTLY this schema:

# {{
#   "mission_id": "{canonical_context.get('mission_id','')}",
#   "initial_state": "Normal",
#   "states": {{
#     "Normal": {{
#       "description": "...",
#       "actions": ["...", "..."]
#     }},
#     "Escalation": {{
#       "description": "...",
#       "actions": ["...", "..."]
#     }},
#     "Alert": {{
#       "description": "...",
#       "actions": ["...", "..."]
#     }},
#     "Inform": {{
#       "description": "...",
#       "actions": ["...", "..."]
#     }}
#     // + 1-4 custom states here (if needed)
#   }},
#   "transitions": [
#     {{
#       "from": "Normal",
#       "to": "Escalation",
#       "condition": "..."
#     }}
#     // more transitions
#   ]
# }}

# FINAL SELF-CHECK (silent; do not output)
# Before you output JSON, verify:
# 1) All 4 core states exist exactly once with correct names
# 2) Total states count is 5-8
# 3) Every condition uses ONLY ALLOWED VARIABLES
# 4) No transition references missing states
# 5) Actions are 2-3 per state, snake_case, plausible from context

# <end_of_turn>
# <start_of_turn>model
# {{"""
    
#     prompt = f"""<start_of_turn>user
# You are creating a state machine for a mission.

# MISSION CONTEXT WITH VARIABLES:
# {json.dumps(canonical_context, indent=2)}

# YOUR TASK:
# 1. Analyze the mission context and variables
# 2. Select the MOST RELEVANT states from the reference states above
# 3. You can also create new states if needed for this specific mission
# 4. Create transitions between states using the key_variables: {', '.join(variable_names)}
# 5. Each state should have 2-3 brief, specific actions
# 6. Keep these states inside the state machine : Normal, Escalation, Alert, Inform

# OUTPUT REQUIREMENTS:
# - Output ONLY valid JSON, nothing else
# - mission_id MUST be: "{canonical_context.get('mission_id', '')}"
# - Select 5-8 most important states for this mission
# - Keep descriptions brief (one sentence)
# - Keep actions minimal (2-3 actions per state, using simple verbs)
# - Create logical transitions using conditions like: "battery_level < 20" or "status == 'ready'"
# - Use EXACTLY this structure:

# {{
#   "mission_id": "{canonical_context.get('mission_id', '')}",
#   "initial_state": "idle",
#   "states": {{
#     "idle": {{
#       "description": "System waiting for commands",
#       "actions": ["monitor_systems", "listen_for_commands"]
#     }},
#     "active": {{
#       "description": "System performing operations",
#       "actions": ["execute_tasks", "update_status"]
#     }}
#   }},
#     "transitions": [
#     {{
#       "from": "idle",
#       "to": "active"
#     }},
#     {{
#       "from": "active",
#       "to": "idle"
#     }}
#   ]
# }}

# IMPORTANT: Output ONLY the JSON object. Do not add explanations, markdown formatting, or any other text.
# <end_of_turn>
# <start_of_turn>model
# {{
# """
    # Increase token limit for overlay generation
    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    
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


def condition_generator(overlay, model, tokenizer):
    """Refine transition conditions using sample parameters"""
    print("\nStarting Condition Generator...")
    
    # 1. Load Parameters
    try:
        with open('sample_output.json', 'r') as f:
            data = json.load(f)
            params = data.get("parameters", {})
    except FileNotFoundError:
        print("sample_output.json not found, skipping condition refinement.")
        return overlay

    # 2. Filter Numeric/Boolean
    valid_params = {}
    for k, v in params.items():
        if isinstance(v, (int, float, bool)) or (isinstance(v, str) and v.lower() in ['true', 'false']):
            valid_params[k] = v
            
    if not valid_params:
        print("No valid numeric/boolean parameters found.")
        return overlay

    print(f"Using parameters for conditions: {list(valid_params.keys())}")
    
    # 3. Refine Transitions - Modify in place to preserve all other fields
    transitions = overlay.get("transitions", [])
    
    # 3. Refine Transitions - Batch Processing
    transitions = overlay.get("transitions", [])
    if not transitions:
        return overlay
        
    print(f"Refining {len(transitions)} transitions in batch...")
    
    # Prepare transition list for prompt
    trans_list_str = ""
    for i, t in enumerate(transitions):
        trans_list_str += f"{i}. {t.get('from')} -> {t.get('to')}\n"
        
    prompt = f"""<start_of_turn>user
Task: Create boolean conditions for the following state transitions based on the available parameters.

Available Variables:
{json.dumps(valid_params, indent=2)}

Transitions:
{trans_list_str}

Instructions:
1. Output a JSON object where keys are the transition indices (0 to {len(transitions)-1}) and values are the python-style boolean conditions (string).
2. Use ONLY the variables listed above.
3. Logic should make sense for moving between the states.
4. If no condition is needed (always true), use "True".
5. Output ONLY the valid JSON object.

Example Format:
{{
  "0": "temp > 50",
  "1": "pressure < 10",
  "2": "True"
}}
<end_of_turn>
<start_of_turn>model
{{"""
    
    # Generate batch conditions
    # Increase token limit for batch response to avoid truncation
    response = generate_text(model, tokenizer, prompt, max_new_tokens=2048).strip()
    
    # Add opening brace if missing
    if not response.strip().startswith('{'):
        response = '{' + response
        
    # Parse response
    valid_conditions = extract_json_from_response(response)
    
    if valid_conditions:
        print(f"✓ Retrieved {len(valid_conditions)} conditions")
        for i, t in enumerate(transitions):
            # Try string index first, then integer if needed
            cond = valid_conditions.get(str(i)) or valid_conditions.get(i)
            
            if cond:
                # Cleanup condition string
                cond = str(cond).strip('"`').replace('```python', '').replace('```', '').strip()
                t["condition"] = cond
                print(f"  [{i}] {t.get('from')} -> {t.get('to')} : {cond}")
            else:
                print(f"  [{i}] No condition generated, defaulting to True")
                t["condition"] = "True"
    else:
        print("✗ Failed to parse batch conditions, defaulting all to True")
        for t in transitions:
            t["condition"] = "True"

    return overlay
        

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
        print(f"  Mission ID: {mission_context.get('mission_id', 'N/A')}")
    except FileNotFoundError:
        print("✗ MissionContext.json not found.")
        sys.exit(1)

    # 3. Process Documents (Chunk & Extract)
    print("\n" + "=" * 60)
    print("Processing Document Chunks (RAG Pipeline)...")
    print("=" * 60)
    
    # Collect all chunks
    all_chunks = []
    
    # Support multiple files in data/ directory
    data_dir = 'data'
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                text = read_document(file_path)
                if text:
                    file_chunks = chunk_text(text)
                    print(f"Loaded {filename}: {len(file_chunks)} chunks")
                    all_chunks.extend(file_chunks)
    
    if not all_chunks:
        print("✗ No document content found in data/")
        # Fallback to empty context or exit?
        # Let's try to proceed if we have a mission summary at least
        # But for now, exit as documents are key
        pass

    # RAG Selection
    mission_summary = mission_context.get("mission_summary", "") + " " + \
                      mission_context.get("location", "") + " " + \
                      str(mission_context.get("operational_hours", ""))
                      
    if all_chunks:
        print(f"\nPerforming RAG retrieval for query: '{mission_summary}'")
        relevant_chunks = get_relevant_chunks(all_chunks, mission_summary, top_k=5)
    else:
        relevant_chunks = []

    aggregated_info = ""
    for i, chunk in enumerate(relevant_chunks):
        print(f" - Analyzing relevant chunk {i+1}...")
        info = extract_security_info(chunk, model, tokenizer)
        if "None" not in info and info.strip():
            aggregated_info += f"\n--- Source Chunk {i+1} ---\n{info}\n"
    
    if not aggregated_info.strip():
        aggregated_info = "No specific security or operational constraints found in documents."
    
    print(f"✓ Aggregated info from {len(relevant_chunks)} relevant chunks")

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
        print(f"  - Constraints: {len(canonical_context.get('constraints', []))}")
    else:
        print("✗ Failed to generate Canonical Context")
        sys.exit(1)

    # 5. Generate Overlay
    print("\n" + "=" * 60)
    print("Step 2: Generating State Machine Overlay")
    print("=" * 60)
    overlay = generate_overlay(canonical_context, model, tokenizer)
    
    if overlay:
        # 6. Apply Condition Generator
        print("\n" + "=" * 60)
        print("Step 3: Refining Conditions with Condition Generator")
        print("=" * 60)
        overlay = condition_generator(overlay, model, tokenizer)
        
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