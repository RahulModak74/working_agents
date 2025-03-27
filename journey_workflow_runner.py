#!/usr/bin/env python3

import os
import sys
import json
import subprocess
from typing import Dict, Any, List, Optional

# Ensure the main directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import CONFIG

def run_agent_with_json_output(agent_name: str, prompt: str, file_path: Optional[str] = None, 
                               schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a single agent with explicit instructions for JSON output"""
    
    # Add explicit instructions for JSON output
    enhanced_prompt = prompt
    if schema:
        enhanced_prompt += "\n\nYou MUST respond with a valid JSON object that exactly follows this schema:"
        enhanced_prompt += f"\n{json.dumps(schema, indent=2)}"
        enhanced_prompt += "\n\nDo not include any explanations outside the JSON structure. Return ONLY the JSON."
    
    # Build the payload
    payload = {
        "model": CONFIG["default_model"],
        "messages": [
            {
                "role": "system",
                "content": "You are a data analysis assistant. Always respond with properly formatted JSON according to the schema provided."
            },
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]
    }
    
    # Add file content if provided
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            payload["messages"].append({
                "role": "user",
                "content": f"Here is the content of {file_path}:\n{file_content}"
            })
    
    # Output file
    output_file = os.path.join(CONFIG["output_dir"], f"{agent_name}_output.txt")
    
    # Convert payload to JSON string for curl
    payload_str = json.dumps(payload).replace("'", "'\\''")
    
    # Build curl command
    curl_command = f"""curl {CONFIG["endpoint"]} \\
  -H "Authorization: Bearer {CONFIG["api_key"]}" \\
  -H "Content-Type: application/json" \\
  -o "{output_file}" \\
  -d '{payload_str}'"""
    
    # Execute the command
    print(f"Running agent {agent_name}...")
    subprocess.run(curl_command, shell=True, check=True)
    
    # Read and parse the response
    with open(output_file, 'r', encoding='utf-8') as f:
        response = f.read()
    
    try:
        # Parse the API response
        json_response = json.loads(response)
        content = json_response.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Try to extract a JSON object from the content
        try:
            # See if the entire content is a JSON object
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            # Try to extract JSON using regex
            import re
            json_pattern = r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}'
            match = re.search(json_pattern, content)
            if match:
                json_str = match.group(1) or content
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    print(f"Could not parse JSON from: {json_str[:100]}...")
                    return {"error": "Failed to parse JSON", "raw_content": content}
            else:
                print(f"No JSON pattern found in: {content[:100]}...")
                return {"error": "No JSON found", "raw_content": content}
    except json.JSONDecodeError:
        print(f"Invalid API response: {response[:100]}...")
        return {"error": "Invalid API response", "raw_content": response}

def run_customer_journey_analysis(workflow_file: str, csv_file: str):
    """Run the customer journey analysis workflow"""
    
    # Load the workflow
    with open(workflow_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # Store results for each agent
    results = {}
    
    # Process each agent in the workflow
    for step in workflow:
        agent_name = step["agent"]
        content = step.get("content", "")
        file_param = step.get("file")
        output_format = step.get("output_format", {})
        schema = output_format.get("schema") if output_format.get("type") == "json" else None
        
        # Check if we need to read from previous agents
        if "readFrom" in step:
            for ref_agent in step["readFrom"]:
                if ref_agent in results:
                    content += f"\n\nOutput from {ref_agent}:\n{json.dumps(results[ref_agent], indent=2)}"
        
        # Special handling for dynamic agent
        if step.get("type") == "dynamic":
            result = run_agent_with_json_output(
                agent_name, 
                step.get("initial_prompt", ""), 
                csv_file if file_param else None,
                schema
            )
            
            # Determine the next action
            selected_focus = None
            if "selected_focus" in result:
                selected_focus = result["selected_focus"]
            elif "action" in result:
                selected_focus = result["action"]
            
            results[agent_name] = result
            print(f"Dynamic agent chose: {selected_focus}")
            
            # Execute the chosen action if valid
            if selected_focus and selected_focus in step.get("actions", {}):
                action = step["actions"][selected_focus]
                action_agent = action.get("agent")
                
                if action_agent:
                    action_content = action.get("content", "")
                    
                    # Add references if needed
                    if "readFrom" in action:
                        for ref_agent in action["readFrom"]:
                            if ref_agent in results:
                                action_content += f"\n\nOutput from {ref_agent}:\n{json.dumps(results[ref_agent], indent=2)}"
                    
                    # Run the action agent
                    action_result = run_agent_with_json_output(
                        action_agent,
                        action_content,
                        csv_file if action.get("file") else None,
                        action.get("output_format", {}).get("schema") if action.get("output_format", {}).get("type") == "json" else None
                    )
                    
                    results[action_agent] = action_result
                    print(f"Action agent {action_agent} completed.")
            
        else:  # Standard agent
            result = run_agent_with_json_output(
                agent_name, 
                content, 
                csv_file if file_param else None,
                schema
            )
            results[agent_name] = result
            print(f"Agent {agent_name} completed.")
    
    # Save the complete results
    output_file = os.path.join(CONFIG["output_dir"], "journey_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Complete analysis saved to {output_file}")
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python journey_workflow_runner.py <workflow_file> <csv_file>")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    if not os.path.exists(workflow_file):
        print(f"Workflow file not found: {workflow_file}")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)
    
    results = run_customer_journey_analysis(workflow_file, csv_file)
    
    # Print a summary of the results
    print("\nAnalysis Summary:")
    for agent, result in results.items():
        print(f"- {agent}: {'Success' if 'error' not in result else 'Error: ' + result['error']}")
