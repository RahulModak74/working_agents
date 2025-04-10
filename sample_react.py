import os
import json
import requests
from react_planning_adapter import ReactWorkflowManager

# Load the configuration settings
from config import CONFIG

def call_openrouter_api(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CONFIG['api_key']}"
    }
    data = {
        "model": "quasar-alpha",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(CONFIG["endpoint"], headers=headers, json=data)
    response_data = response.json()
    return response_data["choices"][0]["message"]["content"]
def create_react_session(self, pattern_id: str, session_id: str = None, problem_description: str = None) -> Dict[str, Any]:
    """Create a new React planning session"""
    if pattern_id not in REACT_PATTERNS:
        return {"error": f"Unknown React pattern: {pattern_id}"}
    
    if pattern_id not in self.react_workflows:
        return {"error": f"Workflow for pattern {pattern_id} not loaded"}
    
    # Generate a session ID if not provided
    if session_id is None:
        session_id = f"react_{pattern_id}_{int(time.time())}_{random.randint(1000, 9999)}"
    
    # Create session data structure
    self.session_data[session_id] = {
        "pattern": pattern_id,
        "workflow": copy.deepcopy(self.react_workflows[pattern_id]),
        "problem_description": problem_description,
        "status": "created",
        "current_stage": 0,
        "results": {},
        "action_history": [],
        "learning_log": [],
        "created_at": time.time()
    }
    
    # Add problem description to workflow steps
    if problem_description:
        for i, step in enumerate(self.session_data[session_id]["workflow"]):
            if "content" in step and "{problem_description}" in step["content"]:
                self.session_data[session_id]["workflow"][i]["content"] = \
                    step["content"].replace("{problem_description}", problem_description)
    
    return {
        "status": "success",
        "session_id": session_id,
        "pattern": pattern_id,
        "description": REACT_PATTERNS[pattern_id]["description"],
        "stages": REACT_PATTERNS[pattern_id]["stages"]
    }
def main():
    # Create a ReactWorkflowManager instance
    manager = ReactWorkflowManager()

    # Define the problem
    problem_description = "Plan a surprise birthday party for a 10-year-old boy who loves dinosaurs."

    # Create a new session using the Multi-Step Reasoning pattern
    session_id = manager.create_react_session("multi_step_reasoning", problem_description=problem_description)["session_id"]

    # Execute the workflow steps
    while True:
        # Get the next step in the workflow
        next_step = manager.get_next_step(session_id)

        if next_step["status"] == "completed":
            break

        # Extract the step details
        step_details = next_step["step_details"]
        agent_name = step_details["agent"]
        prompt = step_details["content"]

        # Call the OpenRouter API to generate a response
        response = call_openrouter_api(prompt)

        # Submit the step result
        manager.submit_step_result(session_id, agent_name, json.loads(response))

    # Get the final session results
    results = manager.get_session_results(session_id)

    # Print the results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
