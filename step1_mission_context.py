import json
import os

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

    # 2. Load Generic Template
    try:
        generic_template = load_json('templates/generic_template.json')
        print(f"Loaded generic template.")
    except FileNotFoundError:
        print("Error: templates/generic_template.json not found.")
        return

    # 3. Merge (Simple implicit understanding logic: just adding user fields to the template)
    mission_context = generic_template.copy()
    mission_context['mission_details'] = user_input
    mission_context['implicit_understanding'] = "Security and Threat Analysis Purpose"
    
    # 4. Save Output
    save_json(mission_context, 'MissionContext.json')
    print("Success: Created MissionContext.json")

if __name__ == "__main__":
    main()
