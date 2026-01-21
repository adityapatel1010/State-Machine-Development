import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_ID = "google/gemma-3-270m-it"

def gemma_3_inference(mission_context, documents_content):
    print(f"Initializing Gemma 3 ({MODEL_ID})...")
    
    # Initialize Model and Tokenizer
    # Note: Running on CPU/MPS specific logic might be needed for Mac, 
    # but generic auto device map is usually safest if 'accelerate' is installed,
    # or just simple load. Since user runs remotely, we assume standard CUDA or CPU.
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            device_map="auto", 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    # Construct Prompt
    prompt = f"""<start_of_turn>user
You are a high-level security mission analyst.
Analyze the following Mission Context and Background Documents.
Identify implicit security needs, threat models, and operational constraints.

## Mission Context
{json.dumps(mission_context, indent=2)}

## Documents
{documents_content}

Output a JSON object ONLY, representing the 'Canonical Mission Context'.
The JSON should include fields like 'derived_security_profile', 'operational_constraints', and 'implicit_context_expansion'.
Ensure the output is valid JSON.
<end_of_turn>
<start_of_turn>model
```json
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Generating response...")
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        temperature=0.2,
        do_sample=True
    )
    
    # Decode and basic parsing
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON part (naive extraction since we primed with ```json)
    try:
        # Look for the JSON block
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            # Attempt to find first { and last }
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            json_str = response_text[start:end]
            
        canonical_context = json.loads(json_str)
        return canonical_context
    except Exception as e:
        print(f"Error parsing JSON output: {e}")
        print(f"Raw Output: {response_text}")
        return {"error": "Failed to generate valid JSON", "raw_output": response_text}

def main():
    print("Step 2: Generating CanonicalMissionContext.json (Gemma 3 Enhanced)...")
    
    # 1. Load MissionContext.json
    try:
        with open('MissionContext.json', 'r') as f:
            mission_context = json.load(f)
            print("Loaded MissionContext.json")
    except FileNotFoundError:
        print("Error: MissionContext.json not found. Run Step 1 first.")
        return

    # 2. Load Documents
    try:
        with open('data/documents.txt', 'r') as f:
            documents = f.read()
            print("Loaded documents")
    except FileNotFoundError:
        print("Error: data/documents.txt not found.")
        return

    # 3. LLM Inference
    canonical_mission_context = gemma_3_inference(mission_context, documents)
    
    # 4. Save Output
    with open('CanonicalMissionContext.json', 'w') as f:
        json.dump(canonical_mission_context, f, indent=2)
    print("Success: Created CanonicalMissionContext.json")

if __name__ == "__main__":
    main()
