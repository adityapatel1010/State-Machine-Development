
import json
import os
import uuid

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def main():
    print("Step 1: Generating MissionContext.json...")
    
    # 1. Load User Input
    try:
        user_input = load_json('input_mission.json')
        print(f"Loaded user input: {user_input}")
    except FileNotFoundError:
        print("Error: input_mission.json not found.")
        return

    # 2. Prepare Mission Context (Directly from Input)
    # No template merging, just use the input directly
    mission_context = user_input.copy()
    
    # Optional: Add implicit context if needed by Step 2 prompt, or remove if not.
    # The previous code added this, keeping it for now to ensure prompt context is rich enough.
    if 'implicit_understanding' not in mission_context:
        mission_context['implicit_understanding'] = "Security and Threat Analysis Purpose"
    
    # 3. Generate Mission ID
    mission_id = f"mission_{uuid.uuid4().hex[:8]}"
    mission_context['mission_id'] = mission_id
    print(f"Generated Mission ID: {mission_id}")
    
    # 4. Save Output
    save_json(mission_context, 'MissionContext.json')
    print("Success: Created MissionContext.json (Direct from Input)")

if __name__ == "__main__":
    main()
