import os
import json
import requests
from react_planning_adapter import ReactWorkflowManager
from cognitive_planning_adapter import CognitiveWorkflowManager

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

def main():
    # Create a CognitiveWorkflowManager instance
    manager = CognitiveWorkflowManager()

    # Define the problem
    problem_description = "Plan a family vacation for a family of four with two teenagers who enjoy outdoor activities and adventure."

    # Create a new session using the Adaptive Reasoning pattern
    session_id = manager.create_cognitive_session("adaptive_cognition", problem_description=problem_description)["session_id"]

    # Execute the workflow steps
    while True:
        # Get the next step in the workflow
        next_step = manager.get_next_step(session_id)

        if next_step["status"] == "completed":
            break

        # Handle dynamic agent steps
        if next_step["status"] == "dynamic_step":
            # Call the OpenRouter API to select the next action
            action_prompt = next_step["prompt"]
            action_response = call_openrouter_api(action_prompt)
            selected_action = json.loads(action_response)["selected_action"]

            # Submit the selected action
            action_result = manager.select_dynamic_action(session_id, selected_action)
            print(f"Selected action: {selected_action}")
            print(f"Action details: {action_result['action_details']}")

            # Get the next step after action selection
            next_step = manager.get_next_step(session_id)

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
