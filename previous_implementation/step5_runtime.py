import json
import re
import os
import glob
from flask import Flask, request, jsonify

# Configuration
DSL_PATH = 'MissionSpecDSL.json'
DEFAULT_STATE = 'Normal'

app = Flask(__name__)

class StateManager:
    def __init__(self):
        self.spec = self.load_spec()
        self.current_state = self.spec["spec"]["initial_state"] if self.spec else DEFAULT_STATE
        self.states = self.spec["spec"]["states"] if self.spec else {}
        self.transitions = self.spec["spec"]["transitions"] if self.spec else []
        self.history = []
        self.alert_remaining = 0
        
        # Priority map: index determines priority (Low -> High)
        # Using list(self.states.keys()) relies on insertion order (Python 3.7+)
        state_names = list(self.states.keys())
        self.priority_map = {name: i for i, name in enumerate(state_names)}
        self.current_tick = 0
        self.last_state_entry_tick = 0
        
    def load_spec(self):
        if not os.path.exists(DSL_PATH):
            print(f"Warning: {DSL_PATH} not found.")
            return None
        with open(DSL_PATH, 'r') as f:
            return json.load(f)

    def reset(self):
        self.current_state = self.spec["spec"]["initial_state"]
        self.history = []
        return self.current_state

    def get_context_from_json(self, input_data):
        """
        Extract variables from nested JSON and string fields.
        1. Flattens 'summary' dict.
        2. Parses 'Key=Value' patterns from strings.
        3. Converts types (int, float, bool).
        """
        context = {}
        
        # 1. Base Strategy: Look inside 'summary' if present, otherwise root
        source = input_data.get('summary', input_data)
        
        # Helper to process a dict
        def process_dict(d):
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool)):
                    # Sanitize key first
                    safe_k = "".join([c if c.isalnum() else "_" for c in k]).strip("_")
                    while "__" in safe_k:
                        safe_k = safe_k.replace("__", "_")
                        
                    # Add direct key
                    context[safe_k] = self.infer_type(v)
                    
                    # If string, check for embedded params like "Category=Normal"
                    if isinstance(v, str):
                        self.parse_embedded_params(v, context)
                elif isinstance(v, dict):
                    process_dict(v) # limited recursion
                    
        process_dict(source)
        return context

    def parse_embedded_params(self, text, context):
        # Regex for Key=Value (simple)
        # Matches: Word=Word or Word=Number
        pattern = r'([a-zA-Z0-9_\-\/]+)\s*=\s*([^;\n]+)'
        matches = re.finditer(pattern, text)
        for m in matches:
            k = m.group(1).strip()
            v = m.group(2).strip()
            
            # Sanitize key
            safe_k = "".join([c if c.isalnum() else "_" for c in k]).strip("_")
            while "__" in safe_k:
                safe_k = safe_k.replace("__", "_")
                
            context[safe_k] = self.infer_type(v)

    def infer_type(self, val):
        if isinstance(val, (int, float, bool)):
            return val
        
        val_str = str(val).strip()
        
        # Boolean
        if val_str.lower() == 'true' or val_str.lower() == 'yes':
            return True
        if val_str.lower() == 'false' or val_str.lower() == 'no':
            return False
            
        # Numeric
        try:
            if '.' in val_str:
                return float(val_str)
            else:
                return int(val_str)
        except ValueError:
            pass
            
        # Remove quotes if present
        return val_str.strip('"\'')

    def check_alert_override(self, context):
        THRESHOLD = 85
        
        # Check if we are currently in a forced alert period
        if self.alert_remaining > 0:
            self.alert_remaining -= 1
            if self.current_state != 'Alert':
                 self.transition_to('Alert')
                 return True, {"from": "Override", "to": "Alert", "condition": f"AlertOverride (Remaining: {self.alert_remaining})"}
            return True, None # Stayed in Alert

        # Check for trigger condition
        triggered = False
        for val in context.values():
            if isinstance(val, (int, float)):
                if val > THRESHOLD:
                    triggered = True
                    break
        
        if triggered:
            self.alert_remaining = 60
            if self.current_state != 'Alert':
                self.transition_to('Alert')
                return True, {"from": "Override", "to": "Alert", "condition": "AlertOverride (Triggered)"}
            
            # If already in Alert, we still activate the override counter
            self.alert_remaining = 60
            return True, None
            
        return False, None

    def evaluate_transitions(self, context):
        self.current_tick += 1
        
        # Check override first
        is_override, override_info = self.check_alert_override(context)
        if is_override:
            if override_info:
                print(f"  [OVERRIDE] {override_info['from']} -> {override_info['to']} ({override_info['condition']})")
                return True, override_info
            print("  [OVERRIDE] Staying in Alert.")
            return False, None

        potential_transitions = [t for t in self.transitions if t["from"] == self.current_state]
        triggered = None
        
        print(f"\nEvaluating transitions for State: {self.current_state}")
        # print(f"Context: {json.dumps(context, indent=2)}")
        
        for t in potential_transitions:
            condition = t.get("condition", "True")
            try:
                # Safe context for eval
                # We add 'context' keys as locals
                # Note: This is a simulation sandbox. 
                # Ideally, use a proper expression parser, but eval is requested implicitly by user constraints.
                if condition == "True":
                    triggered = t
                    break
                    
                # Eval
                # Use a custom dict that returns None for missing keys to avoid NameError
                class SafeContext(dict):
                    def __missing__(self, key):
                        return None
                
                safe_context = SafeContext(context)
                
                # Check if the condition works with the current context
                try:
                    result = eval(condition, {"__builtins__": {}}, safe_context)
                except NameError:
                    # Fallback if somehow SafeContext doesn't catch it (e.g. implicitly in some python versions/evals)
                    # But __missing__ should work for direct lookups.
                    result = False
                    
                if result:
                    print(f"  [MATCH] {t['from']} -> {t['to']} (Cond: {condition})")
                    triggered = t
                    break # Priority determined by list order (First match wins)
                else:
                    print(f"  [FALSE] {t['from']} -> {t['to']} (Cond: {condition})")
            except Exception as e:
                print(f"  [ERROR] Condition '{condition}' failed: {e}")
                
        if triggered:
            # Check for priority-based delay
            current_prio = self.priority_map.get(self.current_state, 0)
            target_prio = self.priority_map.get(triggered["to"], 0)
            
            if target_prio < current_prio:
                # High -> Low transition
                elapsed = self.current_tick - self.last_state_entry_tick
                if elapsed < 30:
                    print(f"  [BLOCKED] Priority drop {self.current_state}({current_prio}) -> {triggered['to']}({target_prio}) blocked. Time in state: {elapsed} ticks < 30 ticks")
                    return False, None

            self.transition_to(triggered["to"])
            return True, triggered
            
        print("  No transition triggered.")
        return False, None

    def transition_to(self, new_state):
        self.history.append({
            "from": self.current_state,
            "to": new_state,
            "timestamp": "now" # In real app, use datetime
        })
        self.current_state = new_state
        self.last_state_entry_tick = self.current_tick

# Global Manager Instance
sm = StateManager()

@app.route('/start', methods=['POST'])
def start_simulation():
    state = sm.reset()
    return jsonify({
        "message": "Simulation started/reset",
        "current_state": state,
        "description": sm.states.get(state, {}).get("description", "")
    })

@app.route('/update', methods=['POST'])
def update_state():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    
    # 1. Extract context
    context = sm.get_context_from_json(data)
    
    # 2. Evaluate State
    transitioned, transition_data = sm.evaluate_transitions(context)
    
    response = {
        "current_state": sm.current_state,
        "description": sm.states.get(sm.current_state, {}).get("description", ""),
        "transitioned": transitioned,
        "context_extracted": context
    }
    
    if transitioned:
        response["transition_details"] = {
            "from": transition_data["from"],
            "to": transition_data["to"],
            "condition": transition_data["condition"]
        }
    
    return jsonify(response)

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "current_state": sm.current_state,
        "history_length": len(sm.history)
    })

if __name__ == '__main__':
    import argparse
    import sys
    
    import argparse
    import sys
    
    import argparse
    import sys
    
    # Check if we should run in API mode (optional flag)
    parser = argparse.ArgumentParser(description=f"State Machine Runtime")
    parser.add_argument('--api', action='store_true', help='Run as Flash API server')
    args = parser.parse_args()

    if args.api:
        # API Mode
        print(f"State Machine Runtime API running on http://0.0.0.0:5000")
        print(f"Initial State: {sm.current_state}")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # Directory Batch Mode
        BLOCKS_DIR = './Factory-4/blocks/*.json'
        files = sorted(glob.glob(BLOCKS_DIR))
        
        if not files:
            print(f"No JSON files found in {BLOCKS_DIR}")
        else:
            print(f"Found {len(files)} files in {BLOCKS_DIR}. Processing sequentially...")
            
            for i, file_name in enumerate(files):
                print(f"\n--- Step {i+1}: Processing {file_name} ---")
                try:
                    with open(file_name, 'r') as f:
                        data = json.load(f)
                        
                    context = sm.get_context_from_json(data)
                    transitioned, transition_data = sm.evaluate_transitions(context)
                    
                    result = {
                        "step": i + 1,
                        "file": file_name,
                        "current_state": sm.current_state,
                        "transition_occurred": transitioned,
                    }
                    if transitioned:
                        result["transition"] = {
                            "from": transition_data["from"],
                            "to": transition_data["to"],
                            "condition": transition_data["condition"]
                        }
                    
                    print(json.dumps(result, indent=2))
                    
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON in {file_name}.")
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
