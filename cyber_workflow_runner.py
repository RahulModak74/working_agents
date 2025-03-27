#!/usr/bin/env python3

import os
import sys
import json
import subprocess
import re
from typing import Dict, Any, List, Optional

# Ensure the main directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from config import CONFIG

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling various formats"""
    # Try parsing the entire content as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find anything that looks like a JSON object
    object_pattern = r'({[\s\S]*?})'
    match = re.search(object_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If all extraction attempts fail, return an error object
    return {"error": "Could not extract valid JSON from response", "text": text[:500]}

def run_agent_with_formatted_output(agent_name: str, prompt: str, file_path: Optional[str] = None, 
                                  output_format: Optional[Dict[str, Any]] = None,
                                  references: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run an agent with explicit output formatting instructions"""
    
    # Determine the output type and schema
    output_type = "text"
    schema = None
    sections = None
    
    if output_format:
        output_type = output_format.get("type", "text")
        schema = output_format.get("schema")
        sections = output_format.get("sections")
    
    # Enhance prompt with format instructions
    enhanced_prompt = prompt
    
    # Add reference information if provided
    if references:
        enhanced_prompt += "\n\n### Reference Information:\n"
        for ref_name, ref_content in references.items():
            enhanced_prompt += f"\n#### Output from {ref_name}:\n"
            if isinstance(ref_content, dict):
                enhanced_prompt += json.dumps(ref_content, indent=2)
            else:
                enhanced_prompt += str(ref_content)
    
    # Add explicit formatting instructions
    if output_type == "json" and schema:
        enhanced_prompt += "\n\n### Response Format Instructions:\n"
        enhanced_prompt += "You MUST respond with a valid JSON object exactly matching this schema:\n"
        enhanced_prompt += f"```json\n{json.dumps(schema, indent=2)}\n```\n"
        enhanced_prompt += "\nReturning properly formatted JSON is CRITICAL. Do not include any explanations or text outside the JSON object."
    elif output_type == "markdown" and sections:
        enhanced_prompt += "\n\n### Response Format Instructions:\n"
        enhanced_prompt += "You MUST format your response as a Markdown document containing these exact sections:\n\n"
        for section in sections:
            enhanced_prompt += f"# {section}\n\n"
        enhanced_prompt += "\nEnsure each section heading uses a single # character followed by the exact section name as listed above."
    
    # Output file
    output_file = os.path.join(CONFIG["output_dir"], f"{agent_name}_output.txt")
    
    # Build the payload
    format_type = "JSON" if output_type == "json" else "markdown"
    payload = {
        "model": CONFIG["default_model"],
        "messages": [
            {
                "role": "system",
                "content": f"You are a security analysis assistant specialized in providing {format_type} outputs. Your responses must strictly follow the format specified in the instructions."
            },
            {
                "role": "user",
                "content": enhanced_prompt
            }
        ]
    }
    
    # Add file content if provided
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                payload["messages"].append({
                    "role": "user",
                    "content": f"Here is the content of {os.path.basename(file_path)}:\n{file_content[:5000]}..."
                })
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
    
    # Convert payload to JSON string for curl
    payload_str = json.dumps(payload).replace("'", "'\\''")
    
    # Build curl command
    curl_command = f"""curl {CONFIG["endpoint"]} \\
  -H "Authorization: Bearer {CONFIG["api_key"]}" \\
  -H "Content-Type: application/json" \\
  -o "{output_file}" \\
  -d '{payload_str}'"""
    
    # Execute the command
    print(f"🤖 Running agent: {agent_name}")
    try:
        subprocess.run(curl_command, shell=True, check=True)
        
        # Read and parse the response
        with open(output_file, 'r', encoding='utf-8') as f:
            response = f.read()
        
        # Parse API response
        try:
            json_response = json.loads(response)
            content = json_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Save the content to the output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Process based on output type
            if output_type == "json":
                result = extract_json_from_text(content)
                print(f"✅ {agent_name} completed")
                return result
            elif output_type == "markdown":
                # Verify sections if required
                if sections:
                    missing_sections = []
                    for section in sections:
                        if not re.search(rf"#\s*{re.escape(section)}", content, re.IGNORECASE):
                            missing_sections.append(section)
                    
                    if missing_sections:
                        print(f"Warning: Missing {len(missing_sections)} required sections in markdown output")
                
                print(f"✅ {agent_name} completed")
                return {"markdown_content": content}
            else:
                print(f"✅ {agent_name} completed")
                return {"text_content": content}
                
        except json.JSONDecodeError:
            print(f"❌ Invalid API response from {agent_name}")
            return {"error": "Invalid API response", "content": response[:500]}
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed for {agent_name}: {e}")
        return {"error": f"Command failed: {str(e)}"}

def run_cyber_security_workflow(workflow_file: str, csv_file: str):
    """Run the cyber security analysis workflow"""
    
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
        
        # Collect references from previous agents
        references = {}
        if "readFrom" in step:
            for ref_name in step["readFrom"]:
                if ref_name == "*":
                    # Include all previous results except this agent
                    for prev_agent, prev_result in results.items():
                        if prev_agent != agent_name and prev_agent not in references:
                            references[prev_agent] = prev_result
                elif ref_name in results:
                    references[ref_name] = results[ref_name]
        
        # Handle dynamic agent
        if step.get("type") == "dynamic":
            initial_prompt = step.get("initial_prompt", "")
            
            result = run_agent_with_formatted_output(
                agent_name,
                initial_prompt,
                csv_file if file_param else None,
                output_format,
                references
            )
            
            results[agent_name] = result
            
            # Determine action from result
            action_key = None
            
            # Try to extract action from various possible fields
            if isinstance(result, dict):
                if "response_action" in result:
                    action_key = result["response_action"]
                elif "action" in result:
                    action_key = result["action"]
                elif "selected_focus" in result:
                    action_key = result["selected_focus"]
            
            # Store the action
            results[f"{agent_name}_action"] = action_key
            
            # Check if action is valid and exists in actions
            if action_key and action_key in step.get("actions", {}):
                action = step["actions"][action_key]
                next_agent_name = action.get("agent")
                
                if next_agent_name:
                    action_content = action.get("content", "")
                    
                    # Collect references for the action
                    action_refs = {}
                    if "readFrom" in action:
                        for ref_name in action["readFrom"]:
                            if ref_name == "*":
                                for prev_agent, prev_result in results.items():
                                    if prev_agent != next_agent_name and prev_agent not in action_refs:
                                        action_refs[prev_agent] = prev_result
                            elif ref_name in results:
                                action_refs[ref_name] = results[ref_name]
                    
                    # Run the next agent
                    action_result = run_agent_with_formatted_output(
                        next_agent_name,
                        action_content,
                        csv_file if action.get("file") else None,
                        action.get("output_format"),
                        action_refs
                    )
                    
                    results[next_agent_name] = action_result
            else:
                print(f"Warning: Dynamic agent couldn't determine a valid action: {action_key}")
        else:
            # Standard agent execution
            result = run_agent_with_formatted_output(
                agent_name,
                content,
                csv_file if file_param else None,
                output_format,
                references
            )
            
            results[agent_name] = result
    
    # Save the complete results
    output_file = os.path.join(CONFIG["output_dir"], "cyber_security_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Complete analysis saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cyber_workflow_runner.py <workflow_file> <csv_file>")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    if not os.path.exists(workflow_file):
        print(f"Workflow file not found: {workflow_file}")
        sys.exit(1)
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)
    
    results = run_cyber_security_workflow(workflow_file, csv_file)
    
    # Print a summary of the results
    print("\nWorkflow completed with results:")
    for agent, result in results.items():
        if "_action" in agent:
            print(f"\n=== {agent} ===")
            print(result)
        elif isinstance(result, dict) and "error" in result:
            print(f"\n=== {agent} ===")
            print(f"Error: {result['error']}")
        else:
            print(f"\n=== {agent} ===")
            if isinstance(result, dict):
                print("✓ Success")
            else:
                print(result)
