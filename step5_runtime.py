import json
import time
import random

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class StateManager:
    def __init__(self, mission_spec):
        self.spec = mission_spec["spec"]
        self.current_state = self.spec["initial_state"]
        self.states = self.spec["states"]
        self.transitions = self.spec["transitions"]
        self.running = True

    def get_available_transitions(self):
        return [t for t in self.transitions if t["from"] == self.current_state]

    def process_event(self, event):
        print(f"\n[Event Received]: {event}")
        possible_transitions = self.get_available_transitions()
        
        for t in possible_transitions:
            # Simple condition evaluator
            # Condition format expected: "event == 'type'"
            condition = t["condition"]
            # Extract expected event from string "event == 'foo'"
            if "event ==" in condition:
                expected_event = condition.split("'")[1]
                if event == expected_event:
                    self.transition_to(t["to"])
                    return

        print(f"No transition triggered for event '{event}' in state '{self.current_state}'.")

    def transition_to(self, new_state_name):
        print(f"!!! Transitioning: {self.current_state} -> {new_state_name}")
        self.current_state = new_state_name
        self.execute_state_actions()

    def execute_state_actions(self):
        state_def = self.states[self.current_state]
        print(f"State: {self.current_state}")
        print(f"  Description: {state_def['description']}")
        print(f"  Actions executing: {', '.join(state_def['actions'])}")

    def run_simulation(self):
        print("Starting State Machine Runtime...")
        self.execute_state_actions()
        
        # Simulated sequence of events
        events_sequence = [
            "wait", 
            "suspicious_movement_detected", # Should go to Investigate
            "false_alarm",                  # Should go back to Patrol
            "suspicious_movement_detected", # Investigate again
            "unauthorized_person_confirmed",# Alert_Operator
            "escalate_command_received"     # Lockdown
        ]

        for event in events_sequence:
            time.sleep(1)
            self.process_event(event)
            if self.current_state == "Lockdown":
                print("\n[Runtime] Lockdown initiated. Terminating simulation.")
                break

def main():
    print("Step 5: Initialization Runtime...")
    
    # 1. Load MissionSpecDSL.json
    try:
        dsl = load_json('MissionSpecDSL.json')
        print("Loaded MissionSpecDSL.json")
    except FileNotFoundError:
        print("Error: MissionSpecDSL.json not found. Run Step 4 first.")
        return

    # 2. Initialize State Manager
    sm = StateManager(dsl)
    
    # 3. Run Simulation
    sm.run_simulation()

if __name__ == "__main__":
    main()
