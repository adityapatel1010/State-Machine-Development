import json
import sys

def validate_overlay(overlay):
    print("Validating Overlay...")
    errors = []
    
    # Check basic structure - Support both nested 'state_machine' and flat structure
    if "state_machine" in overlay:
        sm = overlay["state_machine"]
    else:
        # Flat structure fallback
        sm = overlay
    
    # Check states
    if "states" not in sm or not isinstance(sm["states"], dict):
        errors.append("Invalid or missing 'states' dictionary")
        return False, errors
    
    known_states = set(sm["states"].keys())
    
    # Check initial state
    if "initial_state" not in sm:
        errors.append("Missing 'initial_state'")
    elif sm["initial_state"] not in known_states:
        errors.append(f"Initial state '{sm['initial_state']}' is not defined in states.")
        
    # Check transitions
    if "transitions" in sm:
        for i, t in enumerate(sm["transitions"]):
            if "from" not in t or "to" not in t:
                errors.append(f"Transition {i} missing 'from' or 'to'")
                continue
            
            if t["from"] not in known_states:
                errors.append(f"Transition {i} source state '{t['from']}' undefined")
            if t["to"] not in known_states:
                errors.append(f"Transition {i} target state '{t['to']}' undefined")
                
    if errors:
        return False, errors
    return True, []

def compile_to_dsl(overlay):
    print("Compiling Overlay to MissionSpec DSL...")
    
    # Handle flat vs nested structure
    spec = overlay["state_machine"] if "state_machine" in overlay else overlay
    
    dsl = {
        "metadata": {
            "compiler_version": "1.0.0",
            "source_mission_id": overlay.get("mission_id", "unknown")
        },
        "spec": spec
    }
    
    # Add a checksum or signature mock
    dsl["metadata"]["checksum"] = f"hash_{len(str(dsl['spec']))}"
    
    return dsl

def main():
    print("Step 4: Compiling to MissionSpec DSL...")
    
    # 1. Load OverLay.json
    try:
        with open('OverLay.json', 'r') as f:
            overlay = json.load(f)
            print("Loaded OverLay.json")
    except FileNotFoundError:
        print("Error: OverLay.json not found. Run Step 3 first.")
        return

    # 2. Validate
    is_valid, errors = validate_overlay(overlay)
    if not is_valid:
        print("Validation FAILED:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
    print("Validation PASSED.")
    
    # 3. Compile
    mission_spec_dsl = compile_to_dsl(overlay)
    
    # 4. Save Output
    with open('MissionSpecDSL.json', 'w') as f:
        json.dump(mission_spec_dsl, f, indent=2)
    print("Success: Created MissionSpecDSL.json")

if __name__ == "__main__":
    main()
