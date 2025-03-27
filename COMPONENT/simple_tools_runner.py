#!/usr/bin/env python3

import os
import sys
import json

# Ensure both directories are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
component_dir = os.path.join(current_dir, "COMPONENT")
if component_dir not in sys.path:
    sys.path.insert(0, component_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import necessary modules
from agent import Agent
from dynamic_agent import DynamicAgent
from agent_system import AgentSystem
from config import CONFIG

# Define tool methods directly on the Agent class
def execute_tool(self, tool_id, **kwargs):
    """Execute a tool and capture its output"""
    from tools import execute_tool as exec_tool
    result = exec_tool(tool_id, **kwargs)
    self.history.append({
        "tool": tool_id,
        "params": kwargs,
        "result": result
    })
    return result

def execute_with_tools(self, content, available_tools=None, file_param=None, 
                      read_from=None, memory_id=None, output_format=None):
    """Execute with tools available to the agent"""
    # Add more explicit instructions for output format
    enhanced_content = content
    
    if output_format:
        format_type = output_format.get("type", "")
        if format_type == "json" and "schema" in output_format:
            enhanced_content += "\n\nYour response MUST be a valid JSON object exactly matching this schema:"
            enhanced_content += f"\n{json.dumps(output_format['schema'], indent=2)}"
        elif format_type == "markdown" and "sections" in output_format:
            enhanced_content += "\n\nYour response MUST be in markdown format with these exact section headings:"
            for section in output_format["sections"]:
                enhanced_content += f"\n# {section}"
    
    # Regular execution without tool handling for simplicity
    return self.execute(enhanced_content, file_param, read_from, memory_id, output_format)

# Add methods to Agent class
setattr(Agent, 'execute_tool', execute_tool)
setattr(Agent, 'execute_with_tools', execute_with_tools)

# Create a simplified version for DynamicAgent
def execute_dynamic_with_tools(self, initial_prompt, available_tools=None, file_param=None, 
                              read_from=None, memory_id=None, output_format=None, actions=None):
    """Execute dynamic agent with tools"""
    decision = self.execute_with_tools(initial_prompt, available_tools, file_param, read_from, memory_id, output_format)
    
    # Extract action key
    action_key = None
    if isinstance(decision, dict) and "action" in decision:
        action_key = decision["action"]
    
    if not action_key or not actions or action_key not in actions:
        print("Warning: No valid action determined from response")
        return "unknown", decision
    
    # Handle action
    action = actions[action_key]
    next_agent_name = action.get("agent")
    
    if next_agent_name:
        system = AgentSystem()
        agent = system.get_agent(next_agent_name) or system.create_agent(next_agent_name)
        result = agent.execute_with_tools(
            action.get("content", ""),
            action.get("tools", []),
            action.get("file"),
            action.get("readFrom", []),
            memory_id,
            action.get("output_format")
        )
        return action_key, result
    
    return action_key, decision

setattr(DynamicAgent, 'execute_dynamic_with_tools', execute_dynamic_with_tools)

# Add method to AgentSystem
def execute_sequence_with_tools(self, sequence):
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
            action_key, result = agent.execute_dynamic_with_tools(
                initial_prompt=step.get("initial_prompt", ""),
                available_tools=available_tools,
                file_param=step.get("file"),
                read_from=step.get("readFrom", []),
                memory_id=step.get("memory_id"),
                output_format=step.get("output_format"),
                actions=step.get("actions", {})
            )
            results[agent_name] = result
            results[f"{agent_name}_action"] = action_key
            print(f"✅ {agent_name} completed with action: {action_key}")
        else:
            # Standard agent execution with tools
            print(f"🤖 Running agent: {agent_name}")
            result = agent.execute_with_tools(
                content=step.get("content", ""),
                available_tools=available_tools,
                file_param=step.get("file"),
                read_from=step.get("readFrom", []),
                memory_id=step.get("memory_id"),
                output_format=step.get("output_format")
            )
            results[agent_name] = result
            print(f"✅ {agent_name} completed")
    
    return results

setattr(AgentSystem, 'execute_sequence_with_tools', execute_sequence_with_tools)

# Main function
def run_tool_workflow(workflow_file):
    """Run a workflow with tools"""
    with open(workflow_file, 'r', encoding='utf-8') as f:
        workflow = json.load(f)
    
    system = AgentSystem()
    results = system.execute_sequence_with_tools(workflow)
    
    print("Workflow completed with results:")
    for agent, result in results.items():
        print(f"\n=== {agent} ===")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)

# If run directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_tools_runner.py <workflow_file>")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    run_tool_workflow(workflow_file)
