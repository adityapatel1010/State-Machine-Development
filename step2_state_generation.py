
import json
import os
import torch
import sys
import pandas as pd
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import docx

# Configuration
MODEL_ID = "google/gemma-3-4b-it"
CHUNK_SIZE = 1000  # Characters for rough chunking

# --- Helpers (Reused from previous implementation) ---

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

def chunk_text(text, size=CHUNK_SIZE):
    """Split text into chunks for processing"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def get_relevant_chunks(chunks, query, top_k=5):
    """RAG: Select top_k chunks most relevant to the query"""
    print("Initializing Sentence Transformer for RAG...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Encoding {len(chunks)} chunks...")
    corpus_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Cosine similarity
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    
    # Get top k
    top_results = torch.topk(cos_scores, k=min(top_k, len(chunks)))
    
    relevant_chunks = []
    print("\nTop Relevant Chunks:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(f"  - Score: {score:.4f}")
        relevant_chunks.append(chunks[idx])
        
    return relevant_chunks

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
        do_sample=False,  # Greedy decoding for deterministic output
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # Decode only the NEW tokens
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

def extract_json_from_response(response_text):
    """Extract JSON object from model response"""
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        
        if start != -1 and end != 0:
            json_str = response_text[start:end]
            # Fix trailing commas
            import re
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        else:
            print("No JSON object found (missing braces).")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        return None

# --- Core Logic ---

def generate_states_with_classification(mission_context, model, tokenizer):
    """Generate states and classify them as VLM detectable or not"""
    print("Generating Mission States and Classifications...")
    
#     prompt = f"""
# <start_of_turn>user
# You are a **Universal Mission Decomposer → Mission-Semantic VLM State Mapper**.

# SYSTEM CONTRACT (NON-NEGOTIABLE):
# - Output is used to program a VLM that only returns **0-100 confidence scores** for **mission-semantic states**.
# - You MUST produce states in exactly three categories: **Indicator**, **Risk**, **Anomaly**.
# - VLM does NOT compute: timers/durations/windows, event counts across windows, metric distance/speed/trajectory, calibrated geometry, geofences/maps/zones, schedules, identity verification, role binding ("tracked person", "receiving person", "the two"), “who crossed/entered/exited”, or policy actions. Those are **NON_VLM_REQUIREMENTS**.

# INPUT (Ground Truth):
# {json.dumps(mission_context, indent=2)}

# ========================================================
# ABSOLUTE RULES (MUST FOLLOW)
# ========================================================

# R0) INPUT-FAITHFULNESS (NO FABRICATION / NO NORMALIZATION)
# - Preserve all numbers, units, symbols, ranges, inequalities, and negations EXACTLY as written.
# - Never convert units (e.g., seconds→minutes).

# R1) MISSION-SEMANTIC VISUAL ATOMS ONLY (CRITICAL)
# A mission_semantic_visual_atom MUST be a meaningful, mission-level visual concept, such as:
# - **attribute + entity** (e.g., "red jacket", "visible badge", "weapon-like object", "taped boundary line")
# - **interaction / action / behavior** (e.g., "hands object to another person", "drops package", "pushes person", "aggressive posture")
# - **signage/marker visible** (e.g., "EXIT sign visible", "STAFF ONLY marker visible")
# - **visibility/quality condition** (e.g., "subject occluded", "severe blur", "low light")

# You MUST NOT output generic primitives as atoms or states.

# BANNED AS VISUAL ATOMS AND AS VLM STATES (unless explicitly requested as the mission goal):
# - person, people, human, hand, object, motion, detected, present, bbox, tracking, pose, face
# - snake_case variants (person_detected, hand_detected, object_present, etc.)

# If you see these words in the mission text, you must attach them to mission meaning:
# - "hands object" is valid; "object" alone is invalid.
# - "person wearing red jacket" is valid; "person" alone is invalid.

# R2) VLM STATE NAMING (STRICT)
# - Each state_name MUST be PascalCase and MUST end with exactly one suffix:
#   **Indicator** OR **Risk** OR **Anomaly**.
# - state_name MUST NOT contain: detected, present, bbox, tracking.
# - If you cannot form a mission-semantic state, do NOT create a primitive state. Move it to NON_VLM or mark UNKNOWN.

# R3) HARD BAN: TIME/DISTANCE/COUNT/ZONES/CROSSING INSIDE VLM STATES
# No VLM state_name, vlm_conditions.name, or vlm_conditions.Meaning may include:
# - time/duration/windows: within/for/after + seconds/minutes/hours/days/consecutive frames
# - distance/speed/trajectory: meters/feet/km/miles/mph/kph/trajectory/velocity
# - counting/aggregation: ≥/≤/>/<, "N times", "count within window"
# - crossing/entered/exited/separated-by-X as a VLM state

# These MUST become NON_VLM_REQUIREMENTS.

# R4) TRACKING / ROLE BINDING / "WHO DID WHAT" IS NON_VLM
# If mission text includes: tracked person, same person, receiving person, the two, then/after, follow, separate, cross, enter, exit,
# you MUST create NON_VLM_REQUIREMENTS that:
# - include required_modules = "tracker" (and timer/calibration/roi_polygon if needed)
# - bind roles explicitly in Meaning (e.g., giver vs receiver, subject track vs other track)
# VLM provides atoms; tracker binds them to TrackIDs/roles.

# R5) ZONES / GEOFENCES
# - “danger zone / restricted area / within X meters / inside zone” are NON_VLM unless boundary is purely visible and non-metric.
# - VLM may detect boundary markers/signage/lines as Indicators (e.g., "BoundaryMarkerVisibleIndicator").
# - The event "crossed boundary / entered zone / exited zone" is ALWAYS NON_VLM (tracker + boundary geometry).

# R6) IDENTITY / EMPLOYEE / AUTHORIZATION
# - “employee/authorized/registered” is NON_VLM unless mission explicitly defines a purely visual proxy.
# - VLM can output “badge visible” as Indicator; verifying employee status is NON_VLM and may require external lookup.
# - Do NOT assume OCR/QR is available unless mission explicitly says "read/recognize/decode".

# R7) MODULE SELECTION MUST BE EXPLICITLY JUSTIFIED BY TEXT
# Allowed modules:
# ["tracker","timer","calibration","roi_polygon","map_geo","access_control_db","ocr","qr_decoder","policy_engine","logger"]

# Only include:
# - ocr if text explicitly requires reading text/letters/numbers
# - qr_decoder if mission explicitly requires decoding QR/barcode
# - access_control_db if mission explicitly requires verification via database/lookup
# Otherwise do NOT include these modules.

# R8) DEPENDENCY CORRECTNESS
# - Every non_vlm_requirements.depends_on must reference ONLY vlm_states.state_name entries.
# - No dependency may reference banned primitive states.

# ========================================================
# REQUIRED WORKFLOW (DO IN ORDER)
# ========================================================

# STEP 1 — ATOMIC CLAUSE EXTRACTION (EXHAUSTIVE)
# Split all mission text (intent/goals/phases/rules/constraints) into smallest clauses using:
# if/then/and/or/unless/except/within/for/>/< ≥ ≤ ; , .
# Each clause_text_exact must be an exact substring quote (no paraphrase).

# STEP 2 — PER-CLAUSE CLASSIFICATION (EXHAUSTIVE)
# For each atomic clause, produce:
# - mission_semantic_visual_atoms_found: list of mission-semantic visual atoms (may be empty)
# - non_visual_logic_found: list of non-visual logic items (may be empty)
# - implied_external_modules: list of required modules (may be empty) subject to R7

# IMPORTANT:
# - Mixed clauses MUST populate BOTH lists.
# - Do NOT list primitive nouns in mission_semantic_visual_atoms_found.

# STEP 3 — BUILD VLM STATES (COMPLETE MAPPING)
# For each distinct mission_semantic_visual_atom found across all clauses:
# - Create exactly one VLM state (merge only if identical).
# - Assign state_type = Indicator|Risk|Anomaly based on meaning.
# - Add vlm_conditions with Strong cues + UNK rule grounded in visibility conditions.

# STEP 4 — BUILD NON_VLM_REQUIREMENTS (LOGIC GRAPH)
# For each non-visual logic item, create a NON_VLM requirement with:
# - name: canonical
# - Meaning: faithful mission-level interpretation (include role binding, gating, exceptions)
# - Value: exact numbers/units/symbols copied from the mission text if present; otherwise empty string ""
# - depends_on: list of needed VLM state_names (must exist)
# - required_modules: include tracker/timer/calibration/roi_polygon/etc as implied
# - unknown_when: when computation is unsafe (e.g., no calibration, sign not visible, tracker lost)

# Generic patterns you MUST handle:
# - time windows/durations ("within/for/after X") → timer
# - metric distances ("> X meters/feet") → calibration OR roi_polygon (unknown if missing)
# - crossing/entering/exiting → tracker + roi_polygon boundary event
# - role binding ("tracked person", "receiving person", "the two") → tracker required
# - unless/except/ignore → gating logic requirement referencing relevant VLM states

# STEP 5 — VALIDATION (MUST POPULATE)
# Compute these fields from your own output:
# - invalid_vlm_states: list any VLM state that violates R1/R2/R3 (primitive, time/distance/count, crossing, snake_case, no suffix).
# - visual_atoms_without_states: list any extracted mission_semantic_visual_atom with no matching VLM state.
# - all_atomic_clauses_accounted_for must be false if any clause is not referenced by either a VLM state or a NON_VLM requirement.

# ========================================================
# OUTPUT FORMAT (JSON ONLY)
# ========================================================
# Return ONLY valid JSON matching this schema exactly:

# {{
#   "atomic_clause_extraction": [
#     {{
#       "source_field": "mission_intent|mission_goals|mission_phases|mission_rules|mission_constraints|other",
#       "clause_text_exact": "",
#       "mission_semantic_visual_atoms_found": [""],
#       "non_visual_logic_found": [""],
#       "implied_external_modules": [""]
#     }}
#   ],
#   "vlm_states": [
#     {{
#       "state_name": "",
#       "detectability": "VLM_DETECTABLE",
#       "state_type": "Indicator|Risk|Anomaly",
#       "vlm_conditions": [
#         {{
#           "name": "",
#           "Meaning": "",
#           "Strong cues": "",
#           "UNK rule": "",
#           "confidence_notes": ""
#         }}
#       ]
#     }}
#   ],
#   "non_vlm_requirements": [
#     {{
#       "name": "",
#       "Meaning": "",
#       "Value": "",
#       "depends_on": ["..."],
#       "required_modules": ["..."],
#       "unknown_when": ""
#     }}
#   ],
#   "coverage_check": {{
#     "all_atomic_clauses_accounted_for": true,
#     "unaccounted_clauses": [],
#     "invalid_vlm_states": [],
#     "visual_atoms_without_states": []
#   }}
# }}

# OUTPUT RULES:
# - JSON only. No markdown. No commentary.
# - Must be valid JSON (no truncation).
# - Preserve exact values/units/symbols.
# - Never output banned primitive states.

# <end_of_turn>
# <start_of_turn>model
# """
    prompt = f"""
<start_of_turn>user
You are a **Universal Mission Decomposer → Mission-Semantic VLM State Mapper**.

SYSTEM CONTRACT (NON-NEGOTIABLE):
- Output is used to program a VLM that only returns **0–100 confidence scores** for **mission-semantic states**.
- You MUST produce states in exactly three categories: **Indicator**, **Risk**, **Anomaly**.
- VLM does NOT compute: timers/durations/windows, event counts across windows, metric distance/speed/trajectory, calibrated geometry, geofences/maps/zones, schedules, identity verification, role binding ("tracked person", "receiving person", "the two"), “who crossed/entered/exited”, or policy actions. Those are **NON_VLM_REQUIREMENTS**.

INPUT (Ground Truth):
{json.dumps(mission_context, indent=2)}

========================================================
ABSOLUTE RULES (MUST FOLLOW)
========================================================

R0) INPUT-FAITHFULNESS (NO FABRICATION / NO NORMALIZATION)
- Preserve all numbers, units, symbols, ranges, inequalities, and negations EXACTLY as written.
- Never convert units (e.g., seconds→minutes).

R1) MISSION-SEMANTIC VISUAL ATOMS ONLY (CRITICAL)
A mission_semantic_visual_atom MUST be a meaningful, mission-level visual concept, such as:
- **attribute + entity** (e.g., "red jacket", "visible badge", "weapon-like object", "taped boundary line")
- **interaction / action / behavior** (e.g., "hands object to another person", "drops package", "pushes person", "aggressive posture")
- **signage/marker visible** (e.g., "EXIT sign visible", "STAFF ONLY marker visible")
- **visibility/quality condition** (e.g., "subject occluded", "severe blur", "low light")

You MUST NOT output generic primitives as atoms or states.

BANNED AS VISUAL ATOMS AND AS VLM STATES (unless explicitly requested as the mission goal):
- person, people, human, hand, object, motion, detected, present, bbox, tracking, pose, face
- snake_case variants (person_detected, hand_detected, object_present, etc.)

R2) VLM STATE NAMING (STRICT)
- Each state_name MUST be PascalCase and MUST end with exactly one suffix:
  **Indicator** OR **Risk** OR **Anomaly**.
- state_name MUST NOT contain: detected, present, bbox, tracking.
- If multiple clauses refer to the SAME visual concept, you MUST reuse the SAME state_name.

R3) HARD BAN: TIME/DISTANCE/COUNT/ZONES INSIDE VLM STATES
No VLM state_name, vlm_conditions.name, or vlm_conditions.Meaning may include:
- time/duration/windows: within/for/after + seconds/minutes/hours/days
- counting/aggregation: ≥/≤/>/<, "N times", "count within window"
- zones or crossings

ALL such logic MUST become NON_VLM_REQUIREMENTS.

IMPORTANT ADDITION:
- Any clause containing an explicit duration (e.g., "for 60 seconds") MUST produce a NON_VLM_REQUIREMENT
  with:
  - required_modules including "timer"
  - Value copied EXACTLY (e.g., "60 seconds")
  - depends_on referencing the relevant VLM state(s)
- Time clauses MUST NOT be dropped or absorbed into VLM conditions.

R4) TRACKING / ROLE BINDING IS NON_VLM
If mission text includes: same person, tracked person, ignore X person, unless X,
you MUST create NON_VLM_REQUIREMENTS that gate or filter based on VLM states.

R5) ZONES / GEOFENCES
- “danger zone / restricted area” are NON_VLM unless boundary is purely visible and non-metric.
- Monitoring *within* a zone is NON_VLM logic.
- VLM may only output visible boundary markers/signage if explicitly visual.

R6) IDENTITY / AUTHORIZATION
- “handicap person” is NOT a visual atom unless the mission explicitly defines a visual proxy.
- Treat "ignore handicap person" as NON_VLM gating logic unless a visible proxy is specified.

R7) MODULE SELECTION MUST BE JUSTIFIED BY TEXT
Allowed modules:
["tracker","timer","calibration","roi_polygon","map_geo","access_control_db","ocr","qr_decoder","policy_engine","logger"]

Include "timer" whenever duration, persistence, or temporal gating is present.

R8) DEPENDENCY CORRECTNESS
- Every non_vlm_requirements.depends_on MUST reference existing vlm_states.state_name entries.

========================================================
REQUIRED WORKFLOW (DO IN ORDER)
========================================================

STEP 1 — ATOMIC CLAUSE EXTRACTION (EXHAUSTIVE)
Split all mission text into the smallest clauses.
Duration phrases like "for 60 seconds" MUST be isolated as their own atomic clause.

STEP 2 — PER-CLAUSE CLASSIFICATION
For each atomic clause:
- mission_semantic_visual_atoms_found
- non_visual_logic_found (e.g., duration, ignore, gating)
- implied_external_modules (e.g., timer)

STEP 3 — BUILD VLM STATES
- Create ONE VLM state per unique mission_semantic_visual_atom.
- Reuse identical state_name across all clauses.
- Do NOT encode duration or constraints inside VLM states.

STEP 4 — BUILD NON_VLM_REQUIREMENTS
For each non-visual logic item:
- Duration → timer-based requirement
- Ignore/except → gating requirement
- Value MUST preserve exact text (e.g., "60 seconds")

STEP 5 — VALIDATION
- Ensure every atomic clause maps to either a VLM state or a NON_VLM requirement.

========================================================
OUTPUT FORMAT (JSON ONLY)
========================================================
Return ONLY valid JSON matching this schema exactly:

{
  "atomic_clause_extraction": [...],
  "vlm_states": [...],
  "non_vlm_requirements": [...],
  "coverage_check": {...}
}

OUTPUT RULES:
- JSON only.
- Preserve exact values.
- Do NOT create time-based VLM states.

<end_of_turn>
<start_of_turn>model
"""

    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    print(response)
    
    return extract_json_from_response(response)

def main():
    print("=" * 60)
    print("Step 2: State Generation with VLM Detectability Analysis")
    print("=" * 60)

    # 1. Load Model
    try:
        tokenizer, model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # 2. Load Mission Context (from Step 1)
    # Note: Assuming Step 1 has run and MissionContext.json is in root or we need to find it.
    # The previous files were moved to 'previous_implementation', but correct workflow implies
    # step 1 creates a fresh 'MissionContext.json' in current dir. 
    # If not present, warn user.
    mission_context_path = 'MissionContext.json'
    if not os.path.exists(mission_context_path):
        # Fallback to previous implementation location for data continuity if step 1 wasn't run *now*
        # But safest is to fail or check both.
        if os.path.exists('previous_implementation/MissionContext.json'):
             mission_context_path = 'previous_implementation/MissionContext.json'
             print(f"Using MissionContext from {mission_context_path}")
    
    try:
        with open(mission_context_path, 'r') as f:
            mission_context = json.load(f)
            print(f"✓ Loaded Mission Context for: {mission_context.get('mission_id', 'Unknown')}")
    except FileNotFoundError:
        print("✗ MissionContext.json not found. Please run step1_mission_context.py first.")
        sys.exit(1)

    # 3. RAG Pipeline
    print("\nRetrieving Domain Knowledge...")
    all_chunks = []
    
    # Look for data in 'data' folder (retained in root or moved?)
    # Task said "Move json and txt data files to previous_implementation", but usually 'data' folder stays?
    # Let's check 'data' folder in root first.
    data_dir = 'data'
    # If empty or not existing, check previous_implementation/data? 
    # The user said "put all the other script and temp files inside a folder caled previous_implementation".
    # Assuming 'data' folder structure might still be there or moved. 
    # I will check root 'data' first.
    
    found_docs = False
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path) and not filename.startswith('.'):
                text = read_document(file_path)
                if text:
                    file_chunks = chunk_text(text)
                    print(f"Loaded {filename}: {len(file_chunks)} chunks")
                    all_chunks.extend(file_chunks)
                    found_docs = True

    if not found_docs:
        print("⚠ No documents found in data/. Context will be limited to Mission Purpose.")

    # relevant_chunks = []
    # if all_chunks:
    #     # Use mission summary or keys as query
    #     query = mission_context.get("mission_summary", "") 
    #     # Fallback if specific keys exist
    #     if not query:
    #          query = mission_context.get("implicit_understanding", "Mission Safety and Security")
             
    #     relevant_chunks = get_relevant_chunks(all_chunks, query)
    
    # aggregated_info = "\n\n".join(relevant_chunks) if relevant_chunks else "No external documents provided."

    # 4. Generate States
    print("\n" + "="*30)
    print(" INPUT CONTEXT")
    print("="*30)
    print(f"Mission Context:\n{json.dumps(mission_context, indent=2)}")
    # print(f"\nAggregated Info from Docs:\n{aggregated_info}")
    
    # result = generate_states_with_classification(mission_context, aggregated_info, model, tokenizer)
    result = generate_states_with_classification(mission_context, model, tokenizer)
    
    if not result or "states" not in result:
        print("✗ Failed to generate valid states.")
        sys.exit(1)
        
    # 5. Process and Display Output
    vlm_detectable = []
    non_vlm_detectable = []
    
    for state in result.get("states", []):
        if state.get("detectability") == "VLM_DETECTABLE":
            vlm_detectable.append(state)
        else:
            non_vlm_detectable.append(state)
            
    print("\n" + "="*30)
    print(" OUTPUT: VLM DETECTABLE STATES")
    print("="*30)
    if vlm_detectable:
        for s in vlm_detectable:
            state_name = s.get('state_name', 'Unknown')
            conditions_obj = s.get('conditions', {})
            print(f"• {state_name}")
            
            # Print conditions by category
            for cat in ["phase_conditions", "rule_conditions", "constraint_conditions"]:
                cond_list = conditions_obj.get(cat, [])
                if cond_list:
                   print(f"  [{cat.replace('_', ' ').title()}]")
                   for c in cond_list:
                       print(f"    - {c.get('name')}: {c.get('Meaning', 'N/A')}")
                       print(f"      Strong Cues: {c.get('Strong cues', 'N/A')}")
    else:
        print("None")

    print("\n" + "="*30)
    print(" OUTPUT: NON-VLM DETECTABLE STATES")
    print("="*30)
    if non_vlm_detectable:
        for s in non_vlm_detectable:
            state_name = s.get('state_name', 'Unknown')
            conditions_obj = s.get('conditions', {})
            print(f"• {state_name}")
            
            for cat in ["phase_conditions", "rule_conditions", "constraint_conditions"]:
                cond_list = conditions_obj.get(cat, [])
                if cond_list:
                   print(f"  [{cat.replace('_', ' ').title()}]")
                   for c in cond_list:
                       print(f"    - {c.get('name')}: {c.get('Meaning', 'N/A')}")
    else:
        print("None")

    # Save to file
    with open('MissionStates.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved detailed state data to MissionStates.json")

    # Save segregated states
    with open('vlm_states.json', 'w') as f:
        json.dump(vlm_detectable, f, indent=2)
    print(f"✓ Saved VLM states to vlm_states.json")

    with open('non_vlm_states.json', 'w') as f:
        json.dump(non_vlm_detectable, f, indent=2)
    print(f"✓ Saved Non-VLM states to non_vlm_states.json")

if __name__ == "__main__":
    main()
