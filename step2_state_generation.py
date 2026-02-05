
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

def generate_states_with_classification(mission_context, aggregated_info, model, tokenizer):
    """Generate states and classify them as VLM detectable or not"""
    print("Generating Mission States and Classifications...")
    
    prompt = f"""
<start_of_turn>user
You are a Mission Planner with expertise in Computer Vision and Vision-Language Models (VLMs).

MISSION CONTEXT (Ground Truth):
{json.dumps(mission_context, indent=2)}

RELEVANT DOMAIN KNOWLEDGE:
{aggregated_info}

OBJECTIVE:
Derive the **minimum sufficient set of system/mission states** required to describe this mission at a high level.
- Do NOT enumerate implementation details.
- Do NOT include redundant, overlapping, or trivially derived states.
- Prefer **semantic states** over low-level signals.

TASKS:
1. Identify the **smallest complete set of mutually distinct states** the mission can be in.
2. For EACH state, classify it as:
   - VLM_DETECTABLE → Can be reliably inferred from visual evidence alone.
   - NON_VLM_DETECTABLE → Requires internal signals, metadata, logic, or non-visual sensors.

DETECTABILITY CRITERIA:
- VLM_DETECTABLE:
  - State is visually observable by a human reviewing video frames.
  - Strong visual patterns or objects are present.
- NON_VLM_DETECTABLE:
  - State depends on internal logic, timing, intent, configuration, or invisible conditions.
  - No consistent visual evidence exists.

FOR EACH STATE, PROVIDE:
- name: Short, precise, canonical state name.
- detectability: "VLM_DETECTABLE" or "NON_VLM_DETECTABLE".
- Meaning: What this state represents in mission terms.
- Strong cues:
  - If VLM_DETECTABLE: concrete visual indicators.
  - If NON_VLM_DETECTABLE: write "None (not visually observable)".
- UNK rule:
  - Conditions under which the system should mark this state as UNKNOWN due to ambiguity, missing data, or conflicting cues.

CONSTRAINTS:
- Generate ONLY states that are strictly necessary.
- Avoid hierarchical or sub-states.
- Avoid speculative or rare edge-case states.
- Use consistent terminology across states.

OUTPUT FORMAT:
Return ONLY valid JSON matching this schema exactly:
{
  "states": [
    {
      "name": "",
      "detectability": "VLM_DETECTABLE | NON_VLM_DETECTABLE",
      "Meaning": "",
      "Strong cues": "",
      "UNK rule": ""
    }
  ]
}

Do not include explanations, markdown, or commentary outside the JSON.
<end_of_turn>
<start_of_turn>model
{{
"""

    response = generate_text(model, tokenizer, prompt, max_new_tokens=1024)
    
    # Add opening brace if not present
    if not response.strip().startswith('{'):
        response = '{' + response
        
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

    relevant_chunks = []
    if all_chunks:
        # Use mission summary or keys as query
        query = mission_context.get("mission_summary", "") 
        # Fallback if specific keys exist
        if not query:
             query = mission_context.get("implicit_understanding", "Mission Safety and Security")
             
        relevant_chunks = get_relevant_chunks(all_chunks, query)
    
    aggregated_info = "\n\n".join(relevant_chunks) if relevant_chunks else "No external documents provided."

    # 4. Generate States
    print("\n" + "="*30)
    print(" INPUT CONTEXT")
    print("="*30)
    print(f"Mission Context:\n{json.dumps(mission_context, indent=2)}")
    print(f"\nAggregated Info from Docs:\n{aggregated_info}")
    
    result = generate_states_with_classification(mission_context, aggregated_info, model, tokenizer)
    
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
            print(f"• {s.get('name')}: {s.get('Meaning', s.get('description', 'N/A'))}")
            print(f"  Strong Cues: {s.get('Strong cues', s.get('visual_cues', 'N/A'))}")
            print(f"  UNK Rule: {s.get('UNK rule', 'N/A')}")
    else:
        print("None")

    print("\n" + "="*30)
    print(" OUTPUT: NON-VLM DETECTABLE STATES")
    print("="*30)
    if non_vlm_detectable:
        for s in non_vlm_detectable:
            print(f"• {s.get('name')}: {s.get('Meaning', s.get('description', 'N/A'))}")
            print(f"  UNK Rule: {s.get('UNK rule', 'N/A')}")
    else:
        print("None")

    # Save to file
    with open('MissionStates.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Saved detailed state data to MissionStates.json")

if __name__ == "__main__":
    main()
