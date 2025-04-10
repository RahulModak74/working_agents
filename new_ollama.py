#!/usr/bin/env python3

import sys
import os
import json
import re
import subprocess
import importlib.util
import glob
import requests
from typing import Dict, Any, List, Optional

# Ensure the main directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from config import CONFIG
except ImportError:
    # Create default config if not available
    CONFIG = {
        "api_key": "",  # No API key needed for Ollama
        "endpoint": "http://localhost:11434/api/chat",
        "default_model": "llama3.1",
        "output_dir": os.path.join(current_dir, "agent_outputs"),
        "memory_dir": os.path.join(current_dir, "agent_memory")
    }
    # Ensure output directories exist
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["memory_dir"], exist_ok=True)

# Global tool registry to store all available tools from adapters
GLOBAL_TOOL_REGISTRY = {}

# Function to check if an object is hashable (can be used as dictionary key)
def is_hashable(obj):
    """Check if an object can be used as a dictionary key"""
    try:
        hash(obj)
        return True
    except TypeError:
        return False

def load_all_tool_adapters():
    """Dynamically load all tool adapters from files matching *_adapter.py"""
    # Find adapters in current directory
    adapter_files = glob.glob(os.path.join(current_dir, "*_adapter.py"))
    # Also find adapters in subdirectories (especially the COMPONENT directory)
    adapter_files += glob.glob(os.path.join(current_dir, "*", "*_adapter.py"))
    
    print(f"Found {len(adapter_files)} adapter files: {[os.path.basename(f) for f in adapter_files]}")
    
    for adapter_file in adapter_files:
        adapter_name = os.path.basename(adapter_file)
        module_name = os.path.splitext(adapter_name)[0]
        
        try:
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(module_name, adapter_file)
            adapter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter_module)
            
            # Check if the module has a TOOL_REGISTRY
            if hasattr(adapter_module, 'TOOL_REGISTRY'):
                print(f"Loaded tool adapter: {module_name} with {len(adapter_module.TOOL_REGISTRY)} tools")
                
                # Update the global registry
                for tool_id, tool_handler in adapter_module.TOOL_REGISTRY.items():
                    GLOBAL_TOOL_REGISTRY[tool_id] = tool_handler
            
            # Check if the module has an execute_tool function
            if hasattr(adapter_module, 'execute_tool'):
                print(f"Loaded execute_tool from: {module_name}")
                
        except Exception as e:
            print(f"Error loading adapter {adapter_name}: {e}")

def execute_tool(tool_id: str, **kwargs) -> Any:
    """Execute a tool by its ID with the provided parameters"""
    if tool_id not in GLOBAL_TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_id}"}
    
    try:
        handler = GLOBAL_TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    except Exception as e:
        print(f"Error executing tool {tool_id}: {str(e)}")
        return {"error": str(e)}

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

def call_ollama_api(prompt: str, model: str = None, system_prompt: str = None) -> str:
    """
    Call the Ollama API with the given prompt and return the response.
    
    Args:
        prompt (str): The user prompt to send to the model
        model (str): The model name to use
        system_prompt (str, optional): System prompt to guide the model
        
    Returns:
        str: The model's response text
    """
    url = CONFIG["endpoint"]
    model = model or CONFIG["default_model"]
    
    # Create the messages array
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    try:
        print(f"Calling Ollama API with model: {model}")
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the Ollama response
        ollama_response = response.json()
        
        # Extract the content from the response
        content = ollama_response.get("message", {}).get("content", "")
        return content
    
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return f"Error: {str(e)}"

def run_agent(agent_name: str, prompt: str, file_path: Optional[str] = None, 
              output_format: Optional[Dict[str, Any]] = None,
              references: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    
    # Build the system message
    format_type = "JSON" if output_type == "json" else "markdown"
    system_message = f"You are a specialized assistant handling {format_type} outputs. Your responses must strictly follow the format specified in the instructions."
    
    # Add domain-specific additions to system message
    if "security" in agent_name.lower() or "threat" in agent_name.lower() or "cyber" in prompt.lower():
        system_message = "You are a cybersecurity analysis assistant. " + system_message
    elif "journey" in agent_name.lower() or "customer" in agent_name.lower() or "segment" in prompt.lower():
        system_message = "You are a customer journey analysis assistant. " + system_message
    elif "finance" in agent_name.lower() or "investment" in agent_name.lower() or "portfolio" in prompt.lower():
        system_message = "You are a financial analysis assistant. " + system_message
    
    # Add file content if provided - THIS IS CRUCIAL FOR ANALYZING THE CSV
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # For large files, just include a limited amount
                if len(file_content) > 10000:
                    preview = file_content[:10000] + "...[content truncated]..."
                    enhanced_prompt = f"Here is a preview of the content of {os.path.basename(file_path)}:\n```\n{preview}\n```\n\n" + enhanced_prompt
                else:
                    enhanced_prompt = f"Here is the content of {os.path.basename(file_path)}:\n```\n{file_content}\n```\n\n" + enhanced_prompt
            
            print(f"Successfully loaded file: {file_path}")
        except Exception as e:
            print(f"Warning: Could not read file {file_path}: {e}")
    else:
        if file_path:
            print(f"Warning: File not found: {file_path}")
    
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
    
    # Execute the call to Ollama API
    print(f"🤖 Running agent: {agent_name}")
    try:
        content = call_ollama_api(enhanced_prompt, model=CONFIG["default_model"], system_prompt=system_message)
        
        # Save the content to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Process based on output type
        if output_type == "json":
            result = extract_json_from_text(content)
            if "error" in result:
                print(f"⚠️ Warning: JSON extraction failed for {agent_name}. Content: {content[:100]}...")
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
                
    except Exception as e:
        print(f"❌ Error with {agent_name}: {e}")
        return {"error": f"Error: {str(e)}", "content": str(e)}

def run_universal_workflow(workflow_file: str, data_file: str = None):
    """Run any workflow using the universal approach"""
    
    # Get the directory where the workflow file is located
    workflow_dir = os.path.dirname(os.path.abspath(workflow_file))
    
    # First, load all tool adapters
    load_all_tool_adapters()
    
    # Print available tools for debugging
    print(f"Loaded a total of {len(GLOBAL_TOOL_REGISTRY)} tools from all adapters")
    if GLOBAL_TOOL_REGISTRY:
        print(f"Available tools: {', '.join(sorted(GLOBAL_TOOL_REGISTRY.keys()))}")
    
    # Load the workflow
    with open(workflow_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    # Print data file path for debugging
    if data_file:
        print(f"Using data file: {data_file}")
        if not os.path.exists(data_file):
            print(f"WARNING: Data file does not exist: {data_file}")
    
    # Store results for each agent
    results = {}
    
    # Process each agent in the workflow
    for step in workflow:
        agent_name = step["agent"]
        content = step.get("content", "")
        file_param = step.get("file", False)
        output_format = step.get("output_format", {})
        
        # Use data_file if file_param is true or equal to the data_file name, 
        # otherwise use the specific file if provided
        file_to_use = None
        if file_param is True and data_file:
            file_to_use = data_file
            print(f"Using data file for {agent_name}: {file_to_use}")
        elif isinstance(file_param, str):
            if file_param == os.path.basename(data_file) and data_file:
                file_to_use = data_file
            else:
                # If file_param is a string, use it as the file path
                file_to_use = os.path.join(workflow_dir, file_param) if not os.path.isabs(file_param) else file_param
            print(f"Using specific file for {agent_name}: {file_to_use}")
        
        # Collect references from previous agents
        references = {}
        if "readFrom" in step:
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
        
        # Check if tools are required for this step
        if "tools" in step:
            required_tools = step["tools"]
            missing_tools = [tool for tool in required_tools if tool not in GLOBAL_TOOL_REGISTRY]
            
            if missing_tools:
                print(f"Warning: Missing required tools for {agent_name}: {missing_tools}")
                # Add note about missing tools to the agent prompt
                content += f"\n\nNote: The following tools are not available: {', '.join(missing_tools)}"
            else:
                # Add note about available tools to the agent prompt
                content += f"\n\nYou have access to these tools: {', '.join(required_tools)}"
        
        # Handle Ollama-specific configuration
        if "llm_config" in step:
            llm_config = step.get("llm_config", {})
            if llm_config.get("model"):
                CONFIG["default_model"] = llm_config["model"]
            if llm_config.get("endpoint"):
                CONFIG["endpoint"] = llm_config["endpoint"]
        
        # Handle dynamic agent
        if step.get("type") == "dynamic":
            initial_prompt = step.get("initial_prompt", "")
            
            result = run_agent(
                agent_name,
                initial_prompt,
                file_to_use,
                output_format,
                references
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
                    
                    # Handle Ollama-specific configuration in action
                    if "llm_config" in action:
                        action_llm_config = action.get("llm_config", {})
                        if action_llm_config.get("model"):
                            CONFIG["default_model"] = action_llm_config["model"]
                        if action_llm_config.get("endpoint"):
                            CONFIG["endpoint"] = action_llm_config["endpoint"]
                    
                    # Run the next agent
                    action_result = run_agent(
                        next_agent_name,
                        action_content,
                        file_to_use if action.get("file") else None,
                        action.get("output_format"),
                        action_refs
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
                file_to_use,
                output_format,
                references
            )
            
            results[agent_name] = result
            
            # Store in memory if specified
            if "memory_id" in step:
                memory_path = os.path.join(CONFIG["memory_dir"], f"{step['memory_id']}_{agent_name}.json")
                try:
                    with open(memory_path, 'w') as f:
                        json.dump(result, f, indent=2)
                except Exception as e:
                    print(f"Error storing memory for {agent_name}: {e}")
    
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
            print("Usage: python new_ollama_runner.py --workflow <workflow_file> [data_file]")
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
                    if any(key in result for key in ["score", "risk_patterns", "action", "markdown_content"]):
                        print("✓ Success")
                        # Print a preview of the result 
                        for key in ["score", "risk_patterns", "action"]:
                            if key in result:
                                print(f"- {key}: {result[key]}")
                    else:
                        print("✓ Success but unexpected structure")
                else:
                    print(result)
    else:
        # Start the interactive CLI if available
        try:
            from cli import AgentShell
            AgentShell().cmdloop()
        except ImportError:
            print("Error: Could not import AgentShell from cli")
            print("Usage: python new_ollama_runner.py --workflow <workflow_file> [data_file]")
            sys.exit(1)

if __name__ == "__main__":
    main()
