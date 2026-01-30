import json
import re
import os
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
                    # Add direct key
                    context[k] = self.infer_type(v)
                    
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
            context[k] = self.infer_type(v)

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

    def evaluate_transitions(self, context):
        potential_transitions = [t for t in self.transitions if t["from"] == self.current_state]
        triggered = None
        
        print(f"\nEvaluating transitions for State: {self.current_state}")
        print(f"Context: {json.dumps(context, indent=2)}")
        
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
                result = eval(condition, {"__builtins__": {}}, context)
                if result:
                    print(f"  [MATCH] {t['from']} -> {t['to']} (Cond: {condition})")
                    triggered = t
                    break # Priority determined by list order (First match wins)
                else:
                    print(f"  [FALSE] {t['from']} -> {t['to']} (Cond: {condition})")
            except Exception as e:
                print(f"  [ERROR] Condition '{condition}' failed: {e}")
                
        if triggered:
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
    
    # parser = argparse.ArgumentParser(description=f"State Machine Runtime (API or CLI)")
    # parser.add_argument('--input', type=str, help='Path to VLM JSON file to process (single shot). If omitted, starts Flask API.')
    # args = parser.parse_args()

    file_name="./sample_vlm.json"
    
    if file_name:
        # CLI Mode
        if not os.path.exists(file_name):
            print(f"Error: Input file {file_name} not found.")
            sys.exit(1)
            
        print(f"Running in CLI Mode with input: {file_name}")
        try:
            with open(file_name, 'r') as f:
                data = json.load(f)
                
            # Reuse logic
            context = sm.get_context_from_json(data)
            transitioned, transition_data = sm.evaluate_transitions(context)
            
            result = {
                "initial_state": sm.states.get("initial_state", DEFAULT_STATE), # Note: sm.current_state gets updated
                "final_state": sm.current_state,
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
            print("Error: Invalid JSON in input file.")
        except Exception as e:
            print(f"Error processing file: {e}")
            
    else:
        # API Mode
        print(f"State Machine Runtime API running on http://0.0.0.0:5000")
        print(f"Initial State: {sm.current_state}")
        app.run(host='0.0.0.0', port=5000, debug=True)
