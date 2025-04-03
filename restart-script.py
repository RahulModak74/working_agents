#!/usr/bin/env python3

import sys
import os
import json
import subprocess

def create_partial_workflow(workflow_file, output_file, start_from_agent):
    """
    Create a partial workflow starting from a specific agent
    """
    # Load the workflow file
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)
    
    # Find the index of the agent to start from
    start_index = -1
    for i, agent in enumerate(workflow):
        if agent.get('agent') == start_from_agent:
            start_index = i
            break
    
    if start_index == -1:
        print(f"Agent '{start_from_agent}' not found in workflow")
        return None
    
    # Create a partial workflow starting from the specified agent
    partial_workflow = workflow[start_index:]
    
    # Save the partial workflow
    with open(output_file, 'w') as f:
        json.dump(partial_workflow, f, indent=2)
    
    print(f"Partial workflow saved to: {output_file}")
    return output_file

def run_workflow(workflow_file):
    """
    Run the workflow using new_univ_main.py
    """
    command = [sys.executable, "new_univ_main.py", "--workflow", workflow_file]
    print(f"Running command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing workflow: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nWorkflow execution interrupted by user")
        return 130

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python restart_workflow.py <workflow_file> <start_from_agent>")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    start_from_agent = sys.argv[2]
    
    # Create a filename for the partial workflow
    partial_workflow_file = workflow_file.replace('.json', f'_from_{start_from_agent}.json')
    
    # Create the partial workflow
    partial_workflow_file = create_partial_workflow(workflow_file, partial_workflow_file, start_from_agent)
    if partial_workflow_file:
        # Run the partial workflow
        sys.exit(run_workflow(partial_workflow_file))
    else:
        sys.exit(1)
