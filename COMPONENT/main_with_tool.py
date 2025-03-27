#!/usr/bin/env python3

import sys
import json
import os
import traceback
import re

# Add parent directory to path if running from COMPONENT directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also add the current directory to the path
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add COMPONENT directory to path
component_dir = os.path.join(current_dir, "COMPONENT")
if os.path.exists(component_dir) and component_dir not in sys.path:
    sys.path.insert(0, component_dir)

from agent_system import AgentSystem
from agent import Agent
from dynamic_agent import DynamicAgent
from config import CONFIG

# Try to import tools_adapter first, fall back to original tools if needed
tools_adapter_available = False
tools_available = False

try:
    from tools_adapter import execute_tool as adapter_execute_tool
    print("✅ Tools adapter loaded successfully")
    tools_adapter_available = True
except ImportError:
    print("⚠️ Tools adapter not found, trying original tools module")
    try:
        from tools import execute_tool, TOOL_MANAGER
        print("✅ Original tools module loaded")
        tools_available = True
    except ImportError:
        print("⚠️ Tools module not found, will use mock implementations")


# Mock tool execution when no tools are available
def mock_execute_tool(tool_id, **kwargs):
    """Mock tool execution for when real tools are not available"""
    print(f"[MOCK] Executing tool: {tool_id} with params: {kwargs}")
    
    if tool_id.startswith("http:"):
        return {
            "status": 200,
            "content_type": "application/json",
            "data": {
                "message": f"Mock response for {tool_id}",
                "params": kwargs,
                "success": True
            }
        }
    elif tool_id.startswith("sql:"):
        return {
            "success": True,
            "message": f"Mock SQL operation: {tool_id}",
            "results": [{"id": 1, "name": "Mock data"}]
        }
    elif tool_id.startswith("vector_db:"):
        return {
            "success": True,
            "message": f"Mock vector DB operation: {tool_id}",
            "results": [{"id": "doc1", "text": "Sample document"}]
        }
    else:
        return {
            "error": f"Unknown tool: {tool_id}",
            "message": "This is a mock error response"
        }


# Choose the appropriate execute_tool function
def execute_tool(tool_id, **kwargs):
    """Execute a tool with appropriate implementation"""
    if tools_adapter_available:
        return adapter_execute_tool(tool_id, **kwargs)
    elif tools_available:
        return execute_tool(tool_id, **kwargs)
    else:
        return mock_execute_tool(tool_id, **kwargs)


# Define methods to add to the Agent class
def agent_execute_tool(self, tool_config, **kwargs):
    """Execute a tool and capture its output"""
    # Handle different formats of tool configuration
    tool_id = None
    tool_params = {}
    
    if isinstance(tool_config, str):
        # Simple string format: "tool_id"
        tool_id = tool_config
    elif isinstance(tool_config, dict):
        # Dictionary format: {"id": "tool_id", "params": {...}}
        tool_id = tool_config.get("id")
        tool_params = tool_config.get("params", {})
    else:
        print(f"Invalid tool configuration: {tool_config}")
        return {"error": f"Invalid tool configuration: {tool_config}"}
    
    # Merge explicit kwargs with tool_params (kwargs take precedence)
    merged_params = tool_params.copy()
    merged_params.update(kwargs)
    
    # Execute the tool
    try:
        print(f"Executing tool: {tool_id} with params: {merged_params}")
        result = execute_tool(tool_id, **merged_params)
        status = "success"
    except Exception as e:
        print(f"Tool execution error: {str(e)}")
        result = {"error": str(e)}
        status = "error"
    
    # Initialize history if not present
    if not hasattr(self, 'history'):
        self.history = []
        
    # Record in history
    self.history.append({
        "tool": tool_id,
        "params": merged_params,
        "result": result,
        "status": status
    })
    
    return result


def agent_execute_with_tools(self, content, available_tools=None, file_param=None, 
                      read_from=None, memory_id=None, output_format=None):
    """Execute with tools available to the agent"""
    # Process and normalize tools
    processed_tools = []
    
    if available_tools:
        for tool in available_tools:
            if isinstance(tool, str):
                # Simple string format
                processed_tools.append({"id": tool})
            elif isinstance(tool, dict) and "id" in tool:
                # Dictionary format with possible params
                processed_tools.append(tool)
    
    # Enhance content with tool information
    enhanced_content = content
    
    if processed_tools:
        enhanced_content += "\n\n### Available Tools:\n"
        for tool in processed_tools:
            tool_id = tool["id"]
            params = tool.get("params", {})
            
            enhanced_content += f"\n- Tool: `{tool_id}`"
            if params:
                enhanced_content += "\n  Parameters:"
                for param_name, param_value in params.items():
                    enhanced_content += f"\n    - {param_name}: {param_value}"
        
        # Add instructions for tool usage
        enhanced_content += "\n\nTo use a tool, include this exact syntax in your response:"
        enhanced_content += '\n```\n[TOOL] tool_id\nparam1: value1\nparam2: value2\n[/TOOL]\n```'
        enhanced_content += "\nYou'll receive the tool's output which you should incorporate into your final response."
    
    # Add formatting instructions
    if output_format:
        format_type = output_format.get("type", "")
        if format_type == "json" and "schema" in output_format:
            enhanced_content += "\n\n### Response Format Instructions:\n"
            enhanced_content += "You MUST respond with a valid JSON object exactly matching this schema:\n"
            enhanced_content += f"```json\n{json.dumps(output_format['schema'], indent=2)}\n```\n"
            enhanced_content += "\nReturning properly formatted JSON is CRITICAL. Do not include any explanations outside the JSON object."
        elif format_type == "markdown" and "sections" in output_format:
            enhanced_content += "\n\n### Response Format Instructions:\n"
            enhanced_content += "You MUST format your response as a Markdown document containing these exact sections:\n\n"
            for section in output_format["sections"]:
                enhanced_content += f"# {section}\n\n"
            enhanced_content += "\nEnsure each section heading uses a single # character followed by the exact section name as listed above."
    
    # Capture inputs from other agents if specified
    if read_from:
        enhanced_content = self._process_read_from(enhanced_content, read_from)
    
    # Execute with the enhanced content
    print(f"Executing agent with tools: {[t.get('id') for t in processed_tools if isinstance(t, dict) and 'id' in t]}")
    
    try:
        response = self.execute(enhanced_content, file_param=file_param, memory_id=memory_id, output_format=output_format)
    except Exception as e:
        print(f"Error in execute: {str(e)}")
        response = f"Error executing agent: {str(e)}"
    
    # Store response
    if not hasattr(self, 'last_response'):
        self.last_response = None
    self.last_response = response
    
    # Check if we need to process tool calls (only for text responses)
    if isinstance(response, str) and "[TOOL]" in response:
        print("Tool calls detected in response - processing...")
        try:
            processed_response = self._process_tool_calls(response, processed_tools)
            # Update last_response with processed response
            self.last_response = processed_response
            return processed_response
        except Exception as e:
            print(f"Error processing tool calls: {str(e)}")
            # Return original response if tool processing fails
            return response
    else:
        # If no tool calls or already structured response, return as is
        return response


def agent_process_tool_calls(self, response, available_tools):
    """Process and execute any tool calls in the response"""
    # If response is not a string, return as is
    if not isinstance(response, str):
        return response
    
    # Extract tool call blocks
    tool_pattern = r'\[TOOL\](.*?)\[/TOOL\]'
    tool_matches = re.findall(tool_pattern, response, re.DOTALL)
    
    if not tool_matches:
        return response
    
    modified_response = response
    
    # Process each tool call
    for tool_text in tool_matches:
        # Parse the tool call text to extract tool ID and parameters
        lines = [line.strip() for line in tool_text.strip().split('\n') if line.strip()]
        if not lines:
            continue
        
        # Extract tool ID from first line
        tool_id = lines[0].strip()
        
        # Extract parameters from remaining lines
        params = {}
        for line in lines[1:]:
            if ':' in line:
                key, value = line.split(':', 1)
                param_key = key.strip()
                param_value = value.strip()
                
                # Try to parse JSON values if they look like dictionaries or lists
                if param_value.startswith('{') or param_value.startswith('['):
                    try:
                        param_value = json.loads(param_value)
                    except:
                        pass  # Keep as string if parsing fails
                
                params[param_key] = param_value
        
        # Find matching tool configuration
        tool_config = None
        for tool in available_tools:
            if isinstance(tool, dict) and tool.get("id") == tool_id:
                tool_config = tool
                break
            elif tool == tool_id:
                tool_config = tool_id
                break
        
        if not tool_config:
            print(f"Tool not found: {tool_id}")
            result = {"error": f"Tool not found: {tool_id}"}
        else:
            # Execute the tool
            try:
                print(f"Executing tool: {tool_id} with params: {params}")
                result = self.execute_tool(tool_config, **params)
            except Exception as e:
                print(f"Error executing tool: {e}")
                result = {"error": f"Tool execution failed: {str(e)}"}
        
        # Replace the tool call with the result
        tool_call = f"[TOOL]{tool_text}[/TOOL]"
        result_text = f"[TOOL RESULT]\n{json.dumps(result, indent=2)}\n[/TOOL RESULT]"
        modified_response = modified_response.replace(tool_call, result_text)
    
    # After processing all tool calls, try to generate a final response
    print("All tool calls processed, generating final response")
    
    final_prompt = f"""
Here's the response with tool results:

{modified_response}

Based on these tool results, provide your final response.
"""
    
    # Check if we need to specify JSON output format
    if "Response Format Instructions" in response and "MUST respond with a valid JSON object" in response:
        json_schema_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_schema_match:
            schema_text = json_schema_match.group(1)
            final_prompt += f"""
Your response MUST be a valid JSON object following the schema that was provided. 
Do not include any text outside the JSON object.
"""
    
    # Check if we need to specify markdown with sections
    elif "Response Format Instructions" in response and "MUST format your response as a Markdown document" in response:
        section_matches = re.findall(r'# ([^\n]+)', response)
        if section_matches:
            final_prompt += "\nYour response MUST be formatted as Markdown with these exact section headings:\n"
            for section in section_matches:
                final_prompt += f"\n# {section}"
    
    try:
        # Execute the final prompt
        print("Executing final prompt to generate response...")
        final_response = self.execute(final_prompt)
        return final_response
    except Exception as e:
        print(f"Error generating final response: {e}")
        # Return the modified response with tool results
        return modified_response


def agent_process_read_from(self, content, read_from):
    """Process inputs from other agents"""
    system = AgentSystem()
    enhanced_content = content
    
    for agent_name in read_from:
        if agent_name == "*":
            # Include all results except from this agent
            for name, agent in system.agents.items():
                if name != self.name and hasattr(agent, 'last_response') and agent.last_response:
                    enhanced_content += f"\n\n### Input from {name}:\n{agent.last_response}"
        else:
            # Include specific agent result
            if agent_name in system.agents and hasattr(system.agents[agent_name], 'last_response') and system.agents[agent_name].last_response:
                enhanced_content += f"\n\n### Input from {agent_name}:\n{system.agents[agent_name].last_response}"
    
    return enhanced_content


def dynamic_agent_execute_with_tools(self, initial_prompt, available_tools=None, file_param=None, 
                                    read_from=None, memory_id=None, output_format=None, actions=None):
    """Execute dynamic agent with tools"""
    decision = self.execute_with_tools(initial_prompt, available_tools, file_param, read_from, memory_id, output_format)
    
    # Extract action key
    action_key = None
    if isinstance(decision, dict):
        # Try different common action field names
        for key in ["action", "response_action", "selected_focus", "next_step"]:
            if key in decision and decision[key]:
                action_key = decision[key]
                break
    elif isinstance(decision, str):
        # Try to parse JSON from the string response
        try:
            decision_data = json.loads(decision)
            for key in ["action", "response_action", "selected_focus", "next_step"]:
                if key in decision_data and decision_data[key]:
                    action_key = decision_data[key]
                    break
        except:
            # If parsing fails, look for action mentions in the text
            action_pattern = r'action"?\s*:?\s*["\']?([a-zA-Z_]+)["\']?'
            action_match = re.search(action_pattern, decision)
            if action_match:
                potential_action = action_match.group(1)
                if potential_action in actions:
                    action_key = potential_action
    
    print(f"Detected action key: {action_key}")
    
    if not action_key or not actions or action_key not in actions:
        print(f"Warning: No valid action determined from response. Got: {action_key}")
        return "unknown", decision
    
    # Handle action
    action = actions[action_key]
    next_agent_name = action.get("agent")
    
    if next_agent_name:
        system = AgentSystem()
        agent = system.get_agent(next_agent_name) or system.create_agent(next_agent_name)
        
        # Get tools for the action
        action_tools = action.get("tools", [])
        
        result = agent.execute_with_tools(
            action.get("content", ""),
            action_tools,
            action.get("file"),
            action.get("readFrom", []),
            memory_id,
            action.get("output_format")
        )
        return action_key, result
    
    return action_key, decision


def agent_system_execute_sequence_with_tools(self, sequence):
    """Execute a workflow sequence with tools"""
    results = {}
    
    for step in sequence:
        agent_name = step["agent"]
        agent_type = step.get("type", "standard")
        
        # Create agent if it doesn't exist
        if agent_name not in self.agents:
            self.create_agent(agent_name, CONFIG.get("default_model"), agent_type)
        
        agent = self.agents[agent_name]
        
        # Get available tools
        available_tools = step.get("tools", [])
        
        # Handle dynamic agent differently
        if agent_type == "dynamic" and isinstance(agent, DynamicAgent):
            print(f"🤖 Running dynamic agent: {agent_name}")
            try:
                action_key, result = agent.execute_dynamic_with_tools(
                    initial_prompt=step.get("initial_prompt", ""),
                    available_tools=available_tools,
                    file_param=step.get("file"),
                    read_from=step.get("readFrom", []),
                    memory_id=step.get("memory_id"),
                    output_format=step.get("output_format"),
                    actions=step.get("actions", {})
                )
                # Store in results
                results[agent_name] = result
                results[f"{agent_name}_action"] = action_key
                print(f"✅ {agent_name} completed with action: {action_key}")
            except Exception as e:
                print(f"❌ Error with dynamic agent {agent_name}: {str(e)}")
                results[agent_name] = {"error": str(e)}
                results[f"{agent_name}_action"] = "error"
        else:
            # Standard agent execution with tools
            print(f"🤖 Running agent: {agent_name}")
            try:
                result = agent.execute_with_tools(
                    content=step.get("content", ""),
                    available_tools=available_tools,
                    file_param=step.get("file"),
                    read_from=step.get("readFrom", []),
                    memory_id=step.get("memory_id"),
                    output_format=step.get("output_format")
                )
                # Store in results
                results[agent_name] = result
                print(f"✅ {agent_name} completed")
            except Exception as e:
                print(f"❌ Error with agent {agent_name}: {str(e)}")
                results[agent_name] = {"error": str(e)}
    
    return results


def integrate_tools():
    """Integrate tools with agent classes"""
    print("Integrating tools with agent classes...")
    
    # Add methods to Agent class
    setattr(Agent, 'execute_tool', agent_execute_tool)
    setattr(Agent, 'execute_with_tools', agent_execute_with_tools)
    setattr(Agent, '_process_tool_calls', agent_process_tool_calls)
    setattr(Agent, '_process_read_from', agent_process_read_from)
    
    # Initialize attributes if not present
    if not hasattr(Agent, 'history'):
        setattr(Agent, 'history', [])
    if not hasattr(Agent, 'last_response'):
        setattr(Agent, 'last_response', None)
    
    # Add methods to DynamicAgent class
    setattr(DynamicAgent, 'execute_dynamic_with_tools', dynamic_agent_execute_with_tools)
    
    # Add method to AgentSystem class
    setattr(AgentSystem, 'execute_sequence_with_tools', agent_system_execute_sequence_with_tools)
    
    print("✅ Tool integration completed")
    return True


def main():
    """Main function for tool-enabled workflows"""
    if len(sys.argv) < 3:
        print("Usage: python main_with_tool.py --tool-workflow <workflow_file>")
        sys.exit(1)
    
    command = sys.argv[1]
    if command != "--tool-workflow":
        print("Unknown command. Use --tool-workflow")
        sys.exit(1)
    
    workflow_file = sys.argv[2]
    if not os.path.exists(workflow_file):
        print(f"Workflow file not found: {workflow_file}")
        sys.exit(1)
    
    print(f"Loading workflow from {workflow_file}")
    with open(workflow_file, 'r', encoding='utf-8') as f:
        try:
            workflow = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing workflow file: {e}")
            sys.exit(1)
    
    # Create output directory if needed
    output_dir = CONFIG.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the workflow
    workflow_basename = os.path.basename(workflow_file)
    output_workflow_path = os.path.join(output_dir, f"executed_{workflow_basename}")
    with open(output_workflow_path, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    # Integrate tools with agents
    integrate_tools()
    
    # Create agent system and execute workflow
    system = AgentSystem()
    
    try:
        print(f"Starting workflow execution with {len(workflow)} steps")
        results = system.execute_sequence_with_tools(workflow)
        
        # Save the results
        results_path = os.path.join(output_dir, "workflow_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"Workflow results saved to {results_path}")
        
        # Print results
        print("\nWorkflow completed with results:")
        for agent, result in results.items():
            print(f"\n=== {agent} ===")
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
                
    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
