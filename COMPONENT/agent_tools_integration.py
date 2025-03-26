#!/usr/bin/env python3

import json
import os
from typing import Dict, Any, List, Optional

from agent import Agent
from config import CONFIG
from tools import TOOL_MANAGER, execute_tool


def integrate_tools_with_agent(agent_module_path):
    """
    Add the tools integration to your existing agent.py file.
    This function just provides the code to be added - you'll need to
    manually insert it in the right place in your agent.py file.
    """
    # Check if agent.py exists
    if not os.path.exists(agent_module_path):
        print(f"Error: Agent file {agent_module_path} not found")
        return

    # This code should be added to the Agent class in agent.py
    agent_tools_code = """
    def execute_tool(self, tool_id: str, **kwargs) -> Any:
        """Execute a tool and capture its output"""
        from tools import execute_tool
        
        # Execute the tool
        result = execute_tool(tool_id, **kwargs)
        
        # Record tool execution in history
        self.history.append({
            "tool": tool_id,
            "params": kwargs,
            "result": result
        })
        
        return result
    
    def execute_with_tools(self, content: str, available_tools: List[str] = None, 
                           file_param: Optional[str] = None, 
                           read_from: List[str] = None, memory_id: Optional[str] = None,
                           output_format: Optional[Dict[str, Any]] = None) -> Any:
        """Execute with tools available to the agent"""
        read_from = read_from or []
        
        # Create tool descriptions for available tools
        tool_descriptions = []
        if available_tools:
            from tools import TOOL_REGISTRY
            for tool_id in available_tools:
                if tool_id in TOOL_REGISTRY:
                    handler = TOOL_REGISTRY[tool_id]
                    tool_descriptions.append({
                        "tool_id": tool_id,
                        "description": handler.__doc__,
                        "parameters": handler.__annotations__
                    })
        
        # Add tool descriptions to content
        enhanced_content = content
        if tool_descriptions:
            enhanced_content += "\\n\\nYou have access to the following tools:\\n"
            for tool in tool_descriptions:
                enhanced_content += f"- {tool['tool_id']}: {tool['description']}\\n"
            
            enhanced_content += "\\nTo use a tool, respond with the following format:\\n"
            enhanced_content += "```json\\n{\\n  \\"tool\\": \\"tool_id\\",\\n  \\"params\\": {\\n    \\"param1\\": \\"value1\\",\\n    \\"param2\\": \\"value2\\"\\n  }\\n}\\n```\\n"
        
        # Execute initial prompt to get decision
        response = self.execute(enhanced_content, file_param, read_from, memory_id, output_format)
        
        # Check if response indicates tool usage
        tool_call = None
        if isinstance(response, str):
            # Try to extract tool call from response text
            import re
            tool_call_pattern = r'```json\s*({.*?})\s*```'
            match = re.search(tool_call_pattern, response, re.DOTALL)
            if match:
                try:
                    tool_json = json.loads(match.group(1))
                    if "tool" in tool_json and "params" in tool_json:
                        tool_call = tool_json
                except json.JSONDecodeError:
                    pass
        
        # Execute tool if requested
        if tool_call:
            tool_id = tool_call["tool"]
            params = tool_call["params"]
            
            # Only execute if tool is in the available tools list
            if available_tools is None or tool_id in available_tools:
                tool_result = self.execute_tool(tool_id, **params)
                
                # Create follow-up prompt with tool results
                follow_up = f"{enhanced_content}\\n\\nYou used tool {tool_id} with parameters {json.dumps(params)}.\\n"
                follow_up += f"The tool returned:\\n```\\n{json.dumps(tool_result, indent=2)}\\n```\\n"
                follow_up += "\\nBased on this result, please provide your final response."
                
                # Execute follow-up to get final response
                final_response = self.execute(follow_up, file_param, read_from, memory_id, output_format)
                return final_response
        
        return response"""
    
    print("Code to add to your agent.py file:")
    print(agent_tools_code)
    print("\nAdd this code to the Agent class in agent.py")
    
    # Also provide CLI command additions
    cli_tools_code = """
    def do_tool(self, arg):
        \"\"\"Use a tool: tool <agent_name> <tool_id> param1=value1 param2=value2\"\"\"
        args = arg.split()
        if len(args) < 2:
            print("Usage: tool <agent_name> <tool_id> param1=value1 param2=value2")
            return
        
        agent_name = args[0]
        tool_id = args[1]
        
        # Parse parameters
        params = {}
        for param in args[2:]:
            if '=' in param:
                key, value = param.split('=', 1)
                params[key] = value
        
        # Get the agent
        agent = self.system.get_agent(agent_name)
        if not agent:
            print(f"Agent {agent_name} not found")
            return
        
        # Execute the tool
        try:
            result = agent.execute_tool(tool_id, **params)
            print(f"Tool result:")
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
        except Exception as e:
            print(f"Error executing tool: {e}")
    
    def do_tool_run(self, arg):
        \"\"\"Run agent with tools: tool_run <agent_name> <prompt> [tool:tool1,tool2] [file] [ref:agent1,agent2] [memory:id] [format:json|markdown]\"\"\"
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: tool_run <agent_name> <prompt> [tool:tool1,tool2] [file] [ref:agent1,agent2] [memory:id] [format:json|markdown]")
            return
        
        agent_name = parts[0]
        agent = self.system.get_agent(agent_name) or self.system.create_agent(agent_name)
        
        # Extract tool IDs if present
        tools = []
        if "[tool:" in arg:
            tool_part = arg.split("[tool:")[1]
            if "]" in tool_part:
                tools_text = tool_part.split("]")[0]
                tools = [t.strip() for t in tools_text.split(",")]
        
        # Other parameters (same as do_run)
        file_param = None
        if "[file]" in arg:
            file_parts = arg.split("[file]")[1].split("[")[0].strip()
            file_param = file_parts
        
        refs = []
        if "[ref:" in arg:
            ref_part = arg.split("[ref:")[1]
            if "]" in ref_part:
                ref_text = ref_part.split("]")[0]
                refs = [r.strip() for r in ref_text.split(",")]
        
        memory_id = None
        if "[memory:" in arg:
            memory_part = arg.split("[memory:")[1]
            if "]" in memory_part:
                memory_id = memory_part.split("]")[0].strip()
        
        output_format = None
        if "[format:" in arg:
            format_part = arg.split("[format:")[1]
            if "]" in format_part:
                format_type = format_part.split("]")[0].strip()
                if format_type == "json":
                    output_format = {"type": "json"}
                elif format_type == "markdown":
                    output_format = {"type": "markdown"}
        
        # Extract prompt
        prompt_end = min([
            arg.find(f"[{tag}") if f"[{tag}" in arg else len(arg)
            for tag in ["file", "ref:", "memory:", "format:", "tool:"]
        ])
        prompt = arg.replace(agent_name, "", 1).strip()[:prompt_end].strip()
        
        # Execute with tools
        result = agent.execute_with_tools(prompt, tools, file_param, refs, memory_id, output_format)
        print(f"Result from {agent_name} with tools:")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)"""
    
    print("CLI commands to add to cli.py file:")
    print(cli_tools_code)
    
    # Additions for agent_system.py
    agent_system_code = """
    def execute_sequence_with_tools(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        \"\"\"Execute a sequence with support for tools\"\"\"
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
                print(f"✅ {agent_name} completed with action: {action_key}")
                
                # Store the chosen action in results
                results[f"{agent_name}_action"] = action_key
                
                # If the action has spawned another agent execution, get its results
                for action_data in step.get("actions", {}).values():
                    next_agent = action_data.get("agent")
                    if next_agent and next_agent in self.agents and next_agent not in results:
                        results[next_agent] = self.agents[next_agent].get_output()
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
        
        return results"""

    print("\nAgent system methods to add to agent_system.py:")
    print(agent_system_code)
    
    # Additions for main.py
    main_code = """
    # Modification for main.py to support tool workflows
    # Add this code in the main() function after the if len(sys.argv) > 1 block
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--tool-workflow":
        if len(sys.argv) < 3:
            print("Please provide a workflow file path")
            sys.exit(1)
        
        workflow_file = sys.argv[2]
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        system = AgentSystem()
        try:
            results = system.execute_sequence_with_tools(workflow)
            print("Workflow completed with results:")
            for agent, result in results.items():
                print(f"\\n=== {agent} ===")
                if isinstance(result, dict):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
        except Exception as e:
            print("Workflow error:", e)"""
            
    print("\nMain function modifications for tool workflows:")
    print(main_code)
    
    # Additions for dynamic_agent.py
    dynamic_agent_code = """
    def execute_dynamic_with_tools(self, initial_prompt: str, available_tools: List[str] = None,
                                file_param: Optional[str] = None, read_from: List[str] = None, 
                                memory_id: Optional[str] = None, output_format: Optional[Dict[str, Any]] = None,
                                actions: Dict[str, Dict[str, Any]] = None) -> Tuple[str, Any]:
        \"\"\"Execute with tools and determine the next action\"\"\"
        # First, run the initial prompt to get the decision
        decision = self.execute_with_tools(
            initial_prompt, available_tools, file_param, read_from, memory_id, output_format
        )
        
        # Extract the action from the decision
        action_key = None
        if isinstance(decision, dict) and "action" in decision:
            action_key = decision["action"]
        elif isinstance(decision, str):
            # Try to find an action key in the text
            for key in actions.keys():
                if key.lower() in decision.lower():
                    action_key = key
                    break
        
        if not action_key or action_key not in actions:
            print(f"Warning: Dynamic agent couldn't determine action from: {decision}")
            return "unknown", decision
        
        # Get the action details
        action = actions[action_key]
        next_agent_name = action.get("agent")
        
        # Execute the next agent if specified
        if next_agent_name:
            from agent_system import AgentSystem  # Import here to avoid circular imports
            system = AgentSystem()
            agent = system.get_agent(next_agent_name) or system.create_agent(
                next_agent_name, CONFIG["default_model"]
            )
            
            # Get tools for the next agent if specified
            next_tools = action.get("tools", [])
            
            # Execute the next agent with the specified parameters
            next_result = agent.execute_with_tools(
                action.get("content", ""),
                next_tools,
                action.get("file", None),
                action.get("readFrom", read_from),
                memory_id,
                action.get("output_format", None)
            )
            
            return action_key, next_result
        
        return action_key, decision"""
        
    print("\nDynamic agent method to add to dynamic_agent.py:")
    print(dynamic_agent_code)


if __name__ == "__main__":
    # Default path to agent.py
    agent_path = "agent.py"
    integrate_tools_with_agent(agent_path)
