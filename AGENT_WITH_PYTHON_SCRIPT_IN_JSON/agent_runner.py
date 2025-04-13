#!/usr/bin/env python3

import os
import sys
import json
import re
from typing import Dict, Any, List, Optional

# Import the necessary modules
from tool_manager import tool_manager
from utils import (
    extract_json_from_text, 
    extract_tool_calls, 
    is_hashable, 
    log_api, 
    get_config
)

# Import the call_api function
from call_api import call_api
# Import script_runner functions
from script_runner import run_python_script, import_and_run_function

# Get configuration
CONFIG = get_config()

##EXECUTE SCRIPT
def execute_script(step, workflow_dir, data_file=None):
    """Execute a script defined in a workflow step and return its output"""
    from script_runner import run_python_script, import_and_run_function
    
    script_config = step.get("run_script", {})
    
    if not script_config:
        return None, None
    
    script_path = script_config.get("path")
    if not script_path:
        print(f"⚠️ Warning: Missing script path in run_script configuration")
        return {"error": "Missing script path"}, "Missing script path"
    
    # If path is relative, make it relative to the workflow directory
    if not os.path.isabs(script_path):
        script_path = os.path.join(workflow_dir, script_path)
    
    if not os.path.exists(script_path):
        print(f"⚠️ Warning: Script file not found: {script_path}")
        return {"error": f"Script file not found: {script_path}"}, f"Script file not found: {script_path}"
    
    # Get the function if specified
    function_name = script_config.get("function")
    
    # Get additional arguments
    args = script_config.get("args", [])
    
    # Determine what data file to use (step-specific or workflow-level)
    input_file = script_config.get("input_file", data_file)
    
    print(f"🔧 Executing script: {script_path}")
    
    # Execute the script
    if function_name:
        # Import and run a specific function
        result, output = import_and_run_function(script_path, function_name, input_file)
    else:
        # Run the whole script as a subprocess
        result, output = run_python_script(script_path, input_file, args)
    
    if "error" in result:
        print(f"⚠️ Script execution failed: {result['error']}")
    else:
        print(f"✅ Script execution successful")
        
    return result, output
def run_agent(agent_name: str, prompt: str, file_path: Optional[str] = None, 
              output_format: Optional[Dict[str, Any]] = None,
              references: Optional[Dict[str, Any]] = None,
              required_tools: List[str] = None) -> Dict[str, Any]:
    """Run a single agent with the universal workflow runner"""
    
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
    
    # Add file content if provided
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # For large files, just include a limited amount
                if len(file_content) > 10000:
                    preview = file_content
                    enhanced_prompt += f"\n\nHere is a preview of the content of {os.path.basename(file_path)}:\n```\n{preview}\n```"
                else:
                    enhanced_prompt += f"\n\nHere is the content of {os.path.basename(file_path)}:\n```\n{file_content}\n```"
            
            print(f"Successfully loaded file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
    else:
        if file_path:
            print(f"Warning: File not found: {file_path}")
    
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
    
    # Build the payload
    format_type = "JSON" if output_type == "json" else "markdown"
    system_message = f"You are a specialized assistant handling {format_type} outputs. Your responses must strictly follow the format specified in the instructions."
    
    # Add domain-specific additions to system message
    if "security" in agent_name.lower() or "threat" in agent_name.lower() or "cyber" in prompt.lower():
        system_message = "You are a cybersecurity analysis assistant. " + system_message
    elif "journey" in agent_name.lower() or "customer" in agent_name.lower() or "segment" in prompt.lower():
        system_message = "You are a customer journey analysis assistant. " + system_message
    elif "finance" in agent_name.lower() or "investment" in agent_name.lower() or "portfolio" in prompt.lower():
        system_message = "You are a financial analysis assistant. " + system_message
    
    # Add tool information to prompt if required_tools is provided
    if required_tools:
        available_tools = []
        unavailable_tools = []
        
        for tool_id in required_tools:
            if tool_manager.is_tool_available(tool_id):
                available_tools.append(tool_id)
            else:
                unavailable_tools.append(tool_id)
        
        if available_tools:
            enhanced_prompt += f"\n\nYou have access to these tools: {', '.join(available_tools)}"
            system_message += """
You have access to tools specified in the instructions. To use a tool, format your response like this:

I need to use the tool: $TOOL_NAME
Parameters:
{
  "param1": "value1",
  "param2": "value2"
}

Wait for the tool result before continuing.
"""
        
        if unavailable_tools:
            enhanced_prompt += f"\n\nNote: The following tools are not available: {', '.join(unavailable_tools)}"
    
    # Define conversation for tool usage
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": enhanced_prompt}
    ]
    
    # Output file
    output_file = os.path.join(CONFIG["output_dir"], f"{agent_name}_output.txt")
    final_response = None
    
    # Execute the call with potential tool usage loop
    print(f"🤖 Running agent: {agent_name}")
    try:
        # Initial API call
        api_response = call_api(conversation, CONFIG)
        response_content = api_response
        log_api(agent_name, enhanced_prompt, response_content)
        
        # Save the initial response for debugging
        with open(os.path.join(CONFIG["output_dir"], f"{agent_name}_initial_response.txt"), 'w', encoding='utf-8') as f:
            f.write(response_content)
        
        # Enhanced tool call handling:
        # First check if the response has the LLM trying to make tool calls
        if "I need to use the tool:" in response_content:
            # The model is trying to make tool calls
            print(f"Detected tool call attempts in the response for {agent_name}")
            
            # Process all tool calls in the response
            all_tool_calls = extract_tool_calls(response_content)
            
            if all_tool_calls:
                # We have tool calls to process
                for idx, tool_call in enumerate(all_tool_calls[:5]):  # Limit to 5 tool calls
                    tool_name = tool_call["tool_name"]
                    params = tool_call["params"]
                    
                    print(f"📡 Processing tool call {idx+1}: {tool_name}")
                    
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(tool_name, **params)
                    tool_result_str = json.dumps(tool_result, indent=2)
                    
                    # Replace the tool call with the result in the response
                    tool_call_text = tool_call["full_text"]
                    response_content = response_content.replace(
                        tool_call_text, 
                        f"Tool result for {tool_name}:\n```json\n{tool_result_str}\n```"
                    )
                
                # Now get a new response with the tool results incorporated
                final_prompt = f"Here is the result of executing your tool calls:\n\n{response_content}\n\nBased on these results, please provide your final response."
                
                # Add this to the conversation
                conversation.append({"role": "assistant", "content": api_response})
                conversation.append({"role": "user", "content": final_prompt})
                
                # Get the final response
                final_response = call_api(conversation, CONFIG)
            else:
                # No valid tool calls found, use the original response
                final_response = response_content
        else:
            # The response doesn't contain tool calls, use it directly
            final_response = response_content
        
        # Save the content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_response)
        
        # Process based on output type
        if output_type == "json":
            result = extract_json_from_text(final_response)
            if "error" in result:
                print(f"⚠️ Warning: JSON extraction failed for {agent_name}. Content: {final_response[:100]}...")
            print(f"✅ {agent_name} completed")
            return result
        elif output_type == "markdown":
            # Verify sections if required
            if sections:
                missing_sections = []
                for section in sections:
                    if not re.search(rf"#\s*{re.escape(section)}", final_response, re.IGNORECASE):
                        missing_sections.append(section)
                
                if missing_sections:
                    print(f"Warning: Missing {len(missing_sections)} required sections in markdown output")
            
            print(f"✅ {agent_name} completed")
            return {"markdown_content": final_response}
        else:
            print(f"✅ {agent_name} completed")
            return {"text_content": final_response}
    
    except Exception as e:
        print(f"❌ Error with {agent_name}: {e}")
        return {"error": f"Error: {str(e)}", "content": str(e)}
def run_universal_workflow(workflow_file: str, data_file: str = None):
    """Run any workflow using the universal approach"""
    
    # Get the directory where the workflow file is located
    workflow_dir = os.path.dirname(os.path.abspath(workflow_file))
    
    # First, discover all tools in the current directory and subdirectories
    num_tools = tool_manager.discover_tools()
    print(f"🔧 Auto-discovered {num_tools} tools")
    
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
        
        # Check if this step should run a script
        if "run_script" in step:
            # Execute the script
            script_result, script_output = execute_script(step, workflow_dir, data_file)
            
            # Store the script result
            script_name = f"{agent_name}_script"
            results[script_name] = script_result
            
            # If specified to use script output as content
            if step.get("use_script_output_as_content", False) and script_output:
                # Replace or append to content based on configuration
                if step.get("append_script_output", False):
                    content += "\n\n" + script_output
                else:
                    content = script_output
                    
                print(f"Using script output as content for {agent_name}")
            elif script_result:
                print(f"Script result stored as {script_name}")
        
        # Collect references from previous agents
        references = {}
        if "readFrom" in step:
            # Rest of your code...
            for ref_name in step["readFrom"]:
                # Fix for the unhashable type error: Handle dictionaries and handle wildcard
                if ref_name == "*":
                    # Include all previous results except this agent
                    for prev_agent, prev_result in results.items():
                        if prev_agent != agent_name and prev_agent not in references:
                            references[prev_agent] = prev_result
                # Check if ref_name is hashable before using it as a key
                elif is_hashable(ref_name) and ref_name in results:
                    # The reference is to a previous result
                    references[ref_name] = results[ref_name]
                # Handle the case where ref_name is a dictionary
                elif isinstance(ref_name, dict):
                    # Try to extract a usable key from the dictionary
                    if 'id' in ref_name and is_hashable(ref_name['id']) and ref_name['id'] in results:
                        references[str(ref_name['id'])] = results[ref_name['id']]
                    elif 'agent' in ref_name and is_hashable(ref_name['agent']) and ref_name['agent'] in results:
                        references[str(ref_name['agent'])] = results[ref_name['agent']]
                    elif 'name' in ref_name and is_hashable(ref_name['name']) and ref_name['name'] in results:
                        references[str(ref_name['name'])] = results[ref_name['name']]
                    else:
                        print(f"Warning: Could not resolve reference: {ref_name}")
        
        # Get required tools
        required_tools = step.get("tools", [])
        
        # Handle dynamic agent
        if step.get("type") == "dynamic":
            initial_prompt = step.get("initial_prompt", "")
            
            result = run_agent(
                agent_name,
                initial_prompt,
                data_file if file_param else None,
                output_format,
                references,
                required_tools
            )
            
            results[agent_name] = result
            
            # Determine action from result
            action_key = None
            
            # Try to extract action from various possible fields
            if isinstance(result, dict):
                for key in ["response_action", "action", "selected_focus"]:
                    if key in result:
                        action_key = result[key]
                        break
            
            # Store the action
            action_name = f"{agent_name}_action"
            results[action_name] = action_key
            print(f"🔍 Dynamic agent selected action: {action_key}")
            
            # Check if action is valid and exists in actions
            if action_key and is_hashable(action_key) and action_key in step.get("actions", {}):
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
                            elif is_hashable(ref_name) and ref_name in results:
                                action_refs[ref_name] = results[ref_name]
                            elif isinstance(ref_name, dict):
                                print(f"Warning: Dictionary reference in action: {ref_name}")
                    
                    # Get tools for the next action
                    action_tools = action.get("tools", [])
                    
                    # Run the next agent
                    action_result = run_agent(
                        next_agent_name,
                        action_content,
                        data_file if action.get("file") else None,
                        action.get("output_format"),
                        action_refs,
                        action_tools
                    )
                    
                    results[next_agent_name] = action_result
            elif action_key:
                # Handle non-hashable action_key or action_key not in actions
                if not is_hashable(action_key):
                    print(f"Warning: Dynamic agent selected a non-hashable action: {type(action_key)}")
                else:
                    print(f"Warning: Dynamic agent selected invalid action: {action_key}")
        else:
            # Standard agent execution
            result = run_agent(
                agent_name,
                content,
                data_file if file_param else None,
                output_format,
                references,
                required_tools
            )
            
            results[agent_name] = result
    
    # Save the complete results
    output_file = os.path.join(CONFIG["output_dir"], "workflow_results.json")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Complete analysis saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    return results

def main():
    """Main function"""
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
        if len(sys.argv) < 3:
            print("Usage: python enhanced_auto_runner.py --workflow <workflow_file> [data_file]")
            sys.exit(1)
        
        workflow_file = sys.argv[2]
        
        if not os.path.exists(workflow_file):
            print(f"Workflow file not found: {workflow_file}")
            sys.exit(1)
        
        data_file = None
        if len(sys.argv) > 3:
            data_file = sys.argv[3]
            if not os.path.exists(data_file):
                print(f"Data file not found: {data_file}")
                sys.exit(1)
        
        # Run the workflow
        print(f"Executing workflow: {workflow_file}")
        results = run_universal_workflow(workflow_file, data_file)
        
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
    else:
        # Start the interactive CLI if available
        try:
            from cli import AgentShell
            AgentShell().cmdloop()
        except ImportError:
            print("Error: Could not import AgentShell from cli")
            print("Usage: python enhanced_auto_runner.py --workflow <workflow_file> [data_file]")
            sys.exit(1)

# When this file is run directly, execute main()
if __name__ == "__main__":
    # Ensure the main directory is in the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Also ensure COMPONENT directory is in the path
    component_dir = os.path.join(current_dir, "COMPONENT")
    if os.path.exists(component_dir) and component_dir not in sys.path:
        sys.path.insert(0, component_dir)
        
    main()
