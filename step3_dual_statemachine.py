
import json
import re

class StaticStateMachine:
    def __init__(self, thresholds=None):
        # Default thresholds: Monitor > 30, Alert > 60, Emergency > 90
        self.thresholds = thresholds if thresholds else [30, 60, 90]
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
                        # Handle potential percentage or simple numbers strings
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

        # print(f"DEBUG: Extracted Values: {values}")

        for key, val in values.items():
            severity = 0
            # Check thresholds
            # Emergency (> T3)
            if val > self.thresholds[2]:
                severity = 3
            # Alert (> T2)
            elif val > self.thresholds[1]:
                severity = 2
            # Monitor (> T1)
            elif val > self.thresholds[0]:
                severity = 1
            
            if severity > max_severity:
                max_severity = severity
                trigger = key
                max_val = val
            elif severity == max_severity and val > max_val:
                # Tie-break with higher value
                max_val = val
                trigger = key

        # Persistence / Debounce Logic
        # We only change state if the NEW severity is seen for N consecutive frames
        if max_severity != self.current_state_index:
            if max_severity == self.pending_state_index:
                self.consecutive_count += 1
            else:
                self.pending_state_index = max_severity
                self.consecutive_count = 1
            
            print(f"   -> Pending State: {self.states[max_severity]} (Count: {self.consecutive_count}/{self.REQUIRED_CONSECUTIVE_FRAMES})")
            
            if self.consecutive_count >= self.REQUIRED_CONSECUTIVE_FRAMES:
                self.current_state_index = max_severity
                self.current_state = self.states[max_severity]
                self.trigger_details = {"variable": trigger, "value": max_val} if trigger else None
                print(f"   >>> STATE CHANGE CONFIRMED: {self.current_state}")
                # Reset count ideally, or keep it. 
                # If we keep it, it just stays stable. If input changes, it resets.
        else:
            # Input matches current state, reset pending logic to stable
            self.pending_state_index = max_severity
            self.consecutive_count = 1
            self.trigger_details = {"variable": trigger, "value": max_val} if trigger else None

        return self.current_state, self.trigger_details

class DynamicStateMachine:
    """Placeholder for VLM-based dynamic state machine."""
    def __init__(self):
        self.current_state = "Unknown"
        
    def process_input(self, context):
        # print("DynamicStateMachine: Placeholder processing...")
        return self.current_state

def main():
    print("=== step3_dual_statemachine.py (Persistence Mode) ===")
    
    # 1. Initialize Machines
    static_sm = StaticStateMachine(thresholds=[30, 60, 90])
    
    # 2. Simulate a Sequence of Inputs (1 second each)
    # Scenario: Normal -> Emergency (Spike) -> Emergency (Confirmed) -> Normal (Safe)
    
    simulation_sequence = [
        {"desc": "T=0s (Normal)", "data": {"temp": 20}},
        {"desc": "T=1s (Spike > 90)", "data": {"temp": 95}},  # Count 1
        {"desc": "T=2s (Sustained > 90)", "data": {"temp": 96}},  # Count 2 -> TRANSITION
        {"desc": "T=3s (Sustained > 90)", "data": {"temp": 92}},
        {"desc": "T=4s (Drop to Normal)", "data": {"temp": 25}},  # Count 1
        {"desc": "T=5s (Sustained Normal)", "data": {"temp": 22}}, # Count 2 -> TRANSITION
    ]

    print(f"\nConfiguration: State change requires {static_sm.REQUIRED_CONSECUTIVE_FRAMES} consecutive frames.")

    for i, step in enumerate(simulation_sequence):
        print(f"\n[{step['desc']}] Input: {step['data']}")
        state, details = static_sm.process_input(step['data'])
        print(f"   Current State: {state}")

if __name__ == "__main__":
    main()
