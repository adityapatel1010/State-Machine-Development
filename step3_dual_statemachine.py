
import json
import re

# Global Threshold for Dynamic VLM States
GLOBAL_VLM_THRESHOLD = 80

class StaticStateMachine:
    def __init__(self, thresholds=None, excluded_keys=None):
        # Default thresholds: Monitor > 30, Alert > 60, Emergency > 90
        self.thresholds = thresholds if thresholds else [30, 60, 90]
        self.excluded_keys = set(k.lower() for k in excluded_keys) if excluded_keys else set()
        self.states = ["Normal", "Monitor", "Alert", "Emergency"]
        self.current_state_index = 0  # 0: Normal
        self.current_state = self.states[self.current_state_index]
        self.trigger_details = None
        
        # Persistence tracking
        self.pending_state_index = 0
        self.consecutive_count = 0
        self.REQUIRED_CONSECUTIVE_FRAMES = 2

    def _extract_numeric_values(self, data):
        """Recursively extract all numeric values (int, float) from nested JSON."""
        values = {}
        
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values[k] = v
                elif isinstance(v, str):
                    # Try to parse string as number
                    try:
                        clean_v = v.strip().replace('%', '')
                        if clean_v.replace('.', '', 1).isdigit():
                            values[k] = float(clean_v)
                    except ValueError:
                        pass
                elif isinstance(v, (dict, list)):
                    values.update(self._extract_numeric_values(v))
        
        elif isinstance(data, list):
            for item in data:
                values.update(self._extract_numeric_values(item))
                
        return values

    def process_input(self, data):
        """Process input data and update state based on persistence rules."""
        values = self._extract_numeric_values(data)
        max_severity = 0 # 0: Normal, 1: Monitor, 2: Alert, 3: Emergency
        trigger = None
        max_val = 0

        # Find highest severity based on thresholds
        for key, val in values.items():
            # Skip excluded keys (VLM states)
            if key.lower() in self.excluded_keys:
                continue

            severity = 0
            if val > self.thresholds[2]:
                severity = 3
            elif val > self.thresholds[1]:
                severity = 2
            elif val > self.thresholds[0]:
                severity = 1
            
            if severity > max_severity:
                max_severity = severity
                trigger = key
                max_val = val
            elif severity == max_severity and val > max_val:
                max_val = val
                trigger = key

        # Persistence / Debounce Logic
        if max_severity != self.current_state_index:
            if max_severity == self.pending_state_index:
                self.consecutive_count += 1
            else:
                self.pending_state_index = max_severity
                self.consecutive_count = 1
            
            # print(f"   [Static] Pending State: {self.states[max_severity]} ({self.consecutive_count}/{self.REQUIRED_CONSECUTIVE_FRAMES})")
            
            if self.consecutive_count >= self.REQUIRED_CONSECUTIVE_FRAMES:
                self.current_state_index = max_severity
                self.current_state = self.states[max_severity]
                self.trigger_details = {"variable": trigger, "value": max_val} if trigger else None
                print(f"   [Static] >>> STATE CHANGE CONFIRMED: {self.current_state}")
        else:
            self.pending_state_index = max_severity
            self.consecutive_count = 1
            self.trigger_details = {"variable": trigger, "value": max_val} if trigger else None

        return self.current_state, self.trigger_details

class DynamicStateMachine:
    """Sequential monitors VLM states from vlm_states.json."""
    def __init__(self, states_file='vlm_states.json'):
        self.states = []
        self.current_index = 0
        self.finished = False
        
        try:
            with open(states_file, 'r') as f:
                self.states = json.load(f)
            print(f"[Dynamic] Loaded {len(self.states)} VLM states to monitor.")
        except FileNotFoundError:
            print(f"[Dynamic] Warning: {states_file} not found. Dynamic SM will be inactive.")

    def get_state_names(self):
        """Return list of all VLM state names."""
        return [s.get('name') for s in self.states]

    def _extract_values(self, data):
        """Flatten JSON into a single dict for easy key lookup."""
        values = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, (int, float, str)):
                    values[k] = v
                elif isinstance(v, (dict, list)):
                    values.update(self._extract_values(v))
        elif isinstance(data, list):
            for item in data:
                values.update(self._extract_values(item))
        return values

    def process_input(self, data):
        if not self.states or self.finished:
            return "Complete" if self.finished else "Inactive"

        # Current target state to monitor
        target_state_obj = self.states[self.current_index]
        target_name = target_state_obj.get("name")
        
        # Flatten input to find matching keys
        flat_data = self._extract_values(data)
        
        # Check if target state variable exists and > Threshold
        # We normalize keys to lowercase for robust matching
        found = False
        
        # 1. Exact Name Match in Keys
        for key, val in flat_data.items():
            if key.lower() == target_name.lower():
                # Check value
                try:
                    num_val = 0.0
                    
                    # Handle Boolean (True -> 100, False -> 0)
                    if isinstance(val, bool):
                        num_val = 100.0 if val else 0.0
                    
                    # Handle String or Numbers
                    else:
                        str_val = str(val).strip().lower()
                        
                        # Check explicit string booleans
                        if str_val == 'true':
                            num_val = 100.0
                        elif str_val == 'false':
                            num_val = 0.0
                        else:
                            # Extract number using regex (handles "95%", "Score 80", etc.)
                            match = re.search(r'-?\d+(\.\d+)?', str_val)
                            if match:
                                num_val = float(match.group())
                            else:
                                continue # No number found

                    if num_val > GLOBAL_VLM_THRESHOLD:
                        found = True
                        print(f"   [Dynamic] Detected {target_name} (Value: {val} -> {num_val} > {GLOBAL_VLM_THRESHOLD})")
                        break
                except ValueError:
                    continue  # Parsing failed

        if found:
            # Transition Logic
            self.current_index += 1
            if self.current_index < len(self.states):
                next_state = self.states[self.current_index]['name']
                print(f"   [Dynamic] >>> {target_name} completed and moving to {next_state}")
                return f"Monitoring: {next_state}"
            else:
                self.finished = True
                print(f"   [Dynamic] >>> {target_name} completed. All states finished.")
                return "Mission Complete"
        
        return f"Monitoring: {target_name}"


import argparse
from flask import Flask, request, jsonify

# ... (Existing Classes) ...

def run_server():
    """Run Flask API Server for State Machines."""
    app = Flask(__name__)
    
    print("\nInitializing State Machines for Server...")
    # Initialize Machines
    dynamic_sm = DynamicStateMachine('vlm_states.json')
    excluded_keys = dynamic_sm.get_state_names()
    print(f"Excluding keys from Static SM: {excluded_keys}")
    
    static_sm = StaticStateMachine(thresholds=[30, 60, 90], excluded_keys=excluded_keys)

    @app.route('/process', methods=['POST'])
    def process_frame():
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract summary if present (as per user request)
        if "summary" in data:
            process_data = data["summary"]
        else:
            process_data = data

        # Process Static
        s_state, s_details = static_sm.process_input(process_data)
        
        # Process Dynamic
        d_status = dynamic_sm.process_input(process_data)
        
        response = {
            "processed_data": process_data, # Echo for debugging
            "static_state": s_state,
            "static_details": s_details,
            "dynamic_status": d_status,
            "dynamic_step": dynamic_sm.current_index,
            "dynamic_finished": dynamic_sm.finished
        }
        return jsonify(response)

    print("Starting Flask Server on port 5000...")
    app.run(host='0.0.0.0', port=5000)

def main():
    parser = argparse.ArgumentParser(description="Run Dual State Machine")
    parser.add_argument('--server', action='store_true', help="Run in API Server mode using Flask")
    args = parser.parse_args()

    if args.server:
        run_server()
        return

    print("=== step3_dual_statemachine.py (Block Processing Mode) ===")
    
    # 1. Initialize Machines
    dynamic_sm = DynamicStateMachine('vlm_states.json')
    excluded_keys = dynamic_sm.get_state_names()
    print(f"Excluding keys from Static SM: {excluded_keys}")

    static_sm = StaticStateMachine(thresholds=[30, 60, 90], excluded_keys=excluded_keys)
    
    # 2. Process Blocks from Directory
    import os
    import glob
    
    blocks_dir = "./shooting-1/blocks/"
    if not os.path.exists(blocks_dir):
        print(f"Error: Directory {blocks_dir} not found.")
        return

    # Find and sort block files
    block_files = sorted(glob.glob(os.path.join(blocks_dir, "block_*.json")))
    
    if not block_files:
        print(f"No block files found in {blocks_dir}")
        return

    print(f"\n[Dynamic] Global Threshold: {GLOBAL_VLM_THRESHOLD}")
    print(f"Processing {len(block_files)} blocks...")

    for file_path in block_files:
        try:
            with open(file_path, 'r') as f:
                block_data = json.load(f)
            
            # Extract summary
            if "summary" in block_data:
                input_data = block_data["summary"]
            else:
                input_data = block_data # Fallback
                
            block_name = os.path.basename(file_path)
            print(f"\n[{block_name}] Input (Summary): {input_data}")
            
            # Static
            s_state, _ = static_sm.process_input(input_data)
            print(f"   Static Status:  {s_state}")
            
            # Dynamic
            d_state = dynamic_sm.process_input(input_data)
            print(f"   Dynamic Status: {d_state}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
