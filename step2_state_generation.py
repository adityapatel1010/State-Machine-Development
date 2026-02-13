
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

    prompt = f"""
<start_of_turn>user
You are a **Clause-Driven Mission → Mission-Semantic VLM State Compiler**.

SYSTEM CONTRACT (NON-NEGOTIABLE):
- Your output is used to program a Vision-Language Model (VLM) that outputs ONLY **0-100 confidence scores** for **mission-semantic** states.
- States MUST be exactly one of three types:
  1) **Indicator** (observable attribute/presence/signage/marker/visibility condition)
  2) **Risk** (observable harmful/escalatory/unsafe behavior with credible visual evidence)
  3) **Anomaly** (observable out-of-baseline/unexpected pattern ONLY if explicitly requested by mission wording)
- The VLM does NOT compute: timers/durations/windows, event counts across windows, metric distance/speed/trajectory, calibrated geometry, geofences/maps/zones, schedules, identity/authorization verification, role binding (“tracked person”, “receiving person”, “the two”), “who crossed/entered/exited/separated”, escalation policy, or any action outside the frame. Those are **NON_VLM_REQUIREMENTS**.

INPUT (Ground Truth):
{json.dumps(mission_context, indent=2)}

========================================================
ABSOLUTE RULES (MUST FOLLOW)
========================================================

R0) INPUT-FAITHFULNESS (NO FABRICATION / NO NORMALIZATION)
- Preserve numbers/units/symbols/ranges/inequalities/negations EXACTLY as written.
- NEVER convert units (seconds→minutes, m→ft, etc). Never “round” or “normalize”.

R1) CLAUSE-FIRST COMPILATION (MOST IMPORTANT)
- You MUST process EACH atomic clause independently and produce:
  a) 0..N **mission_semantic_vlm_predicates** (visual, mission-level, NOT primitives)
  b) 0..N **non_vlm_logic_items** (timers, distance, counts, tracking, zones, verification, policy)
  c) **proposed_vlm_state_names** derived ONLY from (a)
  d) **proposed_non_vlm_requirement_names** derived ONLY from (b)
- Only AFTER all clauses are processed may you deduplicate/merge.

R2) MISSION-SEMANTIC VLM PREDICATES ONLY (NO PRIMITIVE DETECTOR OUTPUTS)
A mission_semantic_vlm_predicate MUST be a meaningful phrase of one of these forms:
- attribute + entity (>=2 tokens): “red jacket”, “visible name badge”, “high-visibility vest”
- action/interaction (contains a verb): “hands an object to another person”, “drops a package”, “unsafe reach into cell”
- signage/marker visible: “EXIT sign visible”, “STAFF ONLY boundary marker visible”, “yellow safety tape visible”
- visibility/quality condition: “subject occluded”, “severe blur”, “low light”

BANNED as standalone VLM predicates / states unless explicitly requested as mission goal:
- person, people, human, hand, object, motion, detected, present, bbox, tracking, pose, face
- snake_case variants (person_detected, hand_detected, object_present, etc.)

If a candidate predicate is only a primitive noun OR only 1 token → INVALID; do NOT create a VLM state.

R3) VISUAL EVIDENCE TEST (HARD GATE)
A VLM predicate is VALID ONLY if you can write at least **two concrete pixel-level cues** for it.
- If you can only restate the predicate in “Strong cues” (no concrete cues) → predicate is INVALID.
- INVALID predicates MUST become NON_VLM_REQUIREMENTS (or be decomposed into smaller visual proxies).

R4) OUTCOMES / META-INSTRUCTIONS ARE NOT VLM PREDICATES
These are NEVER VLM states unless explicit visual proxies are provided in the mission:
- “run the line like…”, “be a supervisor”, “stop production”, “create scrap”, “injury potential”, “improve OEE”, “ensure compliance”
- “strong visual evidence”, “do not guess”, “output UNKNOWN”
Handle them as NON_VLM policy requirements (gating/priority), not VLM states.

R5) HARD BAN: TIME / DISTANCE / COUNT / ZONES / CROSSING INSIDE VLM STATES
No VLM state_name, vlm_conditions.name, or Meaning may include:
- time/duration/window: within/for/after + seconds/minutes/hours/days/consecutive frames
- distance/speed/trajectory: meters/feet/km/miles/mph/kph/velocity/trajectory
- counts/aggregation: ≥/≤/>/<, “N times”, “count within window”
- crossing/enter/exit/separate-by-X / returning-to-station as VLM state content
These MUST be NON_VLM_REQUIREMENTS.

R6) DIGITS/UNITS FILTER FOR PREDICATES
- Any substring containing digits or unit symbols (e.g., “2m”, “10 minutes”, “≥3”, “< 5 seconds”, “15%”, “>5 frames”) MUST NOT appear in mission_semantic_vlm_predicates.
- It must appear ONLY in non_vlm_logic_items and in non_vlm_requirements.Value.

R7) TRACKING / ROLE BINDING / “WHO DID WHAT” IS NON_VLM
If a clause implies identity binding or sequencing:
- tracked person / same unit / the receiving person / the two / then / after / returning / repeat
You MUST:
- Put that logic in non_vlm_logic_items
- Create NON_VLM_REQUIREMENTS with required_modules including "tracker" (and timer/calibration/roi_polygon as needed)
VLM provides atoms; tracker binds them to TrackIDs/roles.

R8) ZONES / GEOFENCES
- “danger zone / restricted area / robot cell / within X meters / inside zone” are NON_VLM unless boundary is purely visible and non-metric.
- VLM may detect boundary markers/signage/lines as Indicators.
- “crossed boundary / entered zone / exited zone” is ALWAYS NON_VLM (tracker + boundary geometry).

R9) SEVERITY / PRIORITIZATION IS NON_VLM POLICY
- “classify severity Low/Med/High”, “prioritize safety over quality”, “escalate to High if…” are NON_VLM policy logic.
- VLM can output component states (near-miss risk, evasive reaction indicator), but severity labels and prioritization are NON_VLM.

R10) IDENTITY / AUTHORIZATION VERIFICATION
- “authorized”, “employee”, “maintenance staff” is NON_VLM unless defined purely visually by the mission.
- VLM may output: “name badge visible”, “hi-vis vest visible”.
- Do NOT include ocr/qr_decoder/access_control_db unless explicitly required by mission text.

R11) IGNORE METADATA / NON-MISSION FIELDS
- NEVER generate clauses, predicates, states, or requirements from IDs or metadata such as:
  mission_id, implicit_understanding, internal tags, notes, filenames.
- Use ONLY: mission_intent, mission_goals, mission_phases, mission_rules, mission_constraints, other.

R12) VLM STATE NAMING (STRICT)
- Each VLM state_name MUST be PascalCase and MUST end with exactly one suffix:
  Indicator OR Risk OR Anomaly
- state_name MUST NOT include: detected, present, bbox, tracking.
- Every VLM state must cite its source clause EXACTLY in confidence_notes:
  "source_clause: <exact clause_text_exact>"

R13) ANOMALY IS OPTIONAL AND MUST BE JUSTIFIED
- Only create an Anomaly state if the mission explicitly requests out-of-baseline/unexpected behavior.
- Otherwise, prefer Indicator or Risk.

========================================================
REQUIRED WORKFLOW (DO IN ORDER)
========================================================

STEP 1 — ATOMIC CLAUSE EXTRACTION (EXHAUSTIVE)
Split all mission text from allowed fields into smallest clauses using:
if/then/and/or/unless/except/within/for/>/< ≥ ≤ ; , : ( ) .
Each clause_text_exact must be an exact substring quote (no paraphrase).

STEP 2 — PER-CLAUSE MAPPING (EXHAUSTIVE, NO SHORTCUTS)
For EACH clause, output:
- mission_semantic_vlm_predicates: 0..N (must obey R2/R3/R6)
- non_vlm_logic_items: 0..N (must include time/distance/count/zones/tracking/severity/policy)
- implied_external_modules: subset of allowed modules (must obey R10; only text-justified)
- proposed_vlm_state_names: derived ONLY from mission_semantic_vlm_predicates
- proposed_non_vlm_requirement_names: derived ONLY from non_vlm_logic_items

State suffix selection guidance:
- Indicator: attributes, presence of markers/signage, visibility/occlusion, PPE visible, vest visible, badge visible
- Risk: unsafe interactions, near-miss behavior, unsafe entry, hazardous tool use
- Anomaly: ONLY if explicitly requested as out-of-baseline/unexpected

IMPORTANT:
- If a clause yields NO valid mission_semantic_vlm_predicates, proposed_vlm_state_names must be [].
- Do NOT invent proxies. If mission says “stop production” without proxies, it is NON_VLM policy intent.

STEP 3 — BUILD VLM STATES (UNION + DEDUPE)
- Create vlm_states from the UNION of all proposed_vlm_state_names across clauses.
- For each vlm_state, create at least 1 vlm_conditions entry:
  - Strong cues MUST contain at least two concrete visual cues.
  - UNK rule must specify occlusion/blur/distance/angle/light as appropriate.
  - confidence_notes MUST include exact "source_clause: ..."

STEP 4 — BUILD NON_VLM REQUIREMENTS (UNION + DEDUPE)
For each non_vlm_logic_item, create a non_vlm_requirement with:
- name: canonical
- Meaning: faithful interpretation (include exceptions, unless/ignore, priority/severity logic)
- Value: exact numbers/units/symbols copied (or "" if none)
- depends_on: list of relevant VLM state_names (must exist)
- required_modules: choose from allowed list, include tracker/timer/calibration/roi_polygon/map_geo only when required
- unknown_when: when unsafe (e.g., no calibration, boundary not visible, tracker lost)

Generic patterns you MUST handle:
- within/for/after X time → timer
- ≥/≤/>/< N times within window → timer + counter
- meters/feet distance → calibration OR roi_polygon; UNKNOWN if missing
- crossed/entered/exited/returned/separated → tracker + roi_polygon boundary event
- unless/except/ignore → gating requirement referencing VLM predicates it depends on
- “output UNKNOWN / do not guess” → policy requirement (non-vlm)

STEP 5 — VALIDATION (MUST FILL OUT)
- invalid_vlm_states: list any VLM state that violates R2/R3/R5/R12
- visual_atoms_without_states: any predicate extracted that lacks a VLM state
- all_atomic_clauses_accounted_for must be false if any clause is not represented in either vlm_states or non_vlm_requirements.
- unaccounted_clauses must list any missing clause_text_exact.

========================================================
OUTPUT JSON (ONLY)
========================================================
Return ONLY valid JSON matching this schema exactly:

{{
  "atomic_clause_extraction": [
    {{
      "source_field": "mission_intent|mission_goals|mission_rules|other",
      "clause_text_exact": "",
      "mission_semantic_vlm_predicates": [""],
      "non_vlm_logic_items": [""],
      "implied_external_modules": [""],
      "proposed_vlm_state_names": [""],
      "proposed_non_vlm_requirement_names": [""]
    }}
  ],
  "vlm_states": [
    {{
      "state_name": "",
      "detectability": "VLM_DETECTABLE",
      "state_type": "Indicator|Risk|Anomaly",
      "vlm_conditions": [
        {{
          "name": "",
          "Meaning": "",
          "Strong cues": "",
          "UNK rule": "",
          "confidence_notes": "source_clause: <exact clause_text_exact>"
        }}
      ]
    }}
  ],
  "non_vlm_requirements": [
    {{
      "name": "",
      "Meaning": "",
      "Value": "",
      "depends_on": ["..."],
      "required_modules": ["..."],
      "unknown_when": ""
    }}
  ],
  "coverage_check": {{
    "all_atomic_clauses_accounted_for": true,
    "unaccounted_clauses": [],
    "invalid_vlm_states": [],
    "visual_atoms_without_states": []
  }}
}}

OUTPUT RULES:
- JSON ONLY. No markdown. No commentary.
- Must be valid JSON (no truncation).
- Must not generate VLM states from abstract outcomes/meta-intent/policy.
- Must not include time/distance/count/zone/crossing inside VLM state content.

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
