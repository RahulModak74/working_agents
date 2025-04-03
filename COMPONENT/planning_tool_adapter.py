#!/usr/bin/env python3

import os
import sys
from typing import Dict, Any, List

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from planning_tool import PlanningTool, get_planning_tool
from tools import TOOL_MANAGER, register_tool, TOOL_REGISTRY


# Add planning tool to the tool manager
def initialize_planning_tools():
    """Initialize and register planning tools"""
    
    # Add planning tool to the tool manager class
    def get_planning_tool_method(self, name: str = "default") -> PlanningTool:
        """Get or create a planning tool instance"""
        if not hasattr(self, 'planning_tools'):
            self.planning_tools = {}
            
        if name in self.planning_tools:
            return self.planning_tools[name]
        
        # Create a new planning tool
        config = self.tool_configs.get("planning", {}).get(name, {})
        max_steps = config.get("max_steps", 10)
        verbose = config.get("verbose", False)
        
        planning_tool = get_planning_tool(max_steps, verbose)
        self.planning_tools[name] = planning_tool
        
        return planning_tool
    
    def create_planning_tool_method(self, name: str, max_steps: int = 10, verbose: bool = False) -> PlanningTool:
        """Create a named planning tool"""
        if not hasattr(self, 'planning_tools'):
            self.planning_tools = {}
            
        planning_tool = get_planning_tool(max_steps, verbose)
        self.planning_tools[name] = planning_tool
        
        # Save configuration
        if "planning" not in self.tool_configs:
            self.tool_configs["planning"] = {}
        
        self.tool_configs["planning"][name] = {
            "max_steps": max_steps,
            "verbose": verbose
        }
        self._save_configs()
        
        return planning_tool
    
    # Add methods to the tool manager
    setattr(TOOL_MANAGER.__class__, 'get_planning_tool', get_planning_tool_method)
    setattr(TOOL_MANAGER.__class__, 'create_planning_tool', create_planning_tool_method)
    
    # Update get_available_tools method to include planning tools
    original_get_available_tools = TOOL_MANAGER.get_available_tools
    
    def new_get_available_tools(self) -> Dict[str, List[str]]:
        """Get information about all available tools including planning tools"""
        result = original_get_available_tools(self)
        
        # Add planning tools
        if hasattr(self, 'planning_tools'):
            result["planning"] = list(self.planning_tools.keys())
        else:
            result["planning"] = []
            
        return result
    
    setattr(TOOL_MANAGER.__class__, 'get_available_tools', new_get_available_tools)
    
    # Register planning tool functions
    
    # Chain of Thought
    def planning_chain_of_thought(name: str = "default", problem: str = "", steps: List[str] = None) -> Dict[str, Any]:
        """Create a chain-of-thought reasoning prompt for a problem"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.chain_of_thought(problem, steps)
    
    register_tool("planning", "chain_of_thought", planning_chain_of_thought)
    
    # ReAct approach
    def planning_react(name: str = "default", task: str = "", tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a ReAct (Reasoning + Acting) prompt for a task"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.react(task, tools)
    
    register_tool("planning", "react", planning_react)
    
    # Parse ReAct response
    def planning_parse_react(name: str = "default", text: str = "") -> Dict[str, Any]:
        """Parse a ReAct response from an agent"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.extract_reasoning_and_actions(text)
    
    register_tool("planning", "parse_react", planning_parse_react)
    
    # Create Plan
    def planning_create_plan(name: str = "default", goal: str = "", subtasks: List[str] = None) -> Dict[str, Any]:
        """Create a structured plan with a goal and subtasks"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.create_plan(goal, subtasks)
    
    register_tool("planning", "create_plan", planning_create_plan)
    
    # Parse agent plan
    def planning_parse_plan(name: str = "default", plan_text: str = "") -> Dict[str, Any]:
        """Parse a plan created by an agent in free text format"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.parse_agent_plan(plan_text)
    
    register_tool("planning", "parse_plan", planning_parse_plan)
    
    # Execute subtask
    def planning_execute_subtask(name: str = "default", subtask_index: int = 0, result: Any = None) -> Dict[str, Any]:
        """Execute a subtask in the current plan"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.execute_subtask(subtask_index, result)
    
    register_tool("planning", "execute_subtask", planning_execute_subtask)
    
    # Get plan summary
    def planning_get_summary(name: str = "default") -> Dict[str, Any]:
        """Get a summary of the current plan"""
        planning_tool = TOOL_MANAGER.get_planning_tool(name)
        return planning_tool.get_plan_summary()
    
    register_tool("planning", "get_summary", planning_get_summary)
    
    print("✅ Planning tools registered successfully")


# Initialize planning tools when this module is imported
initialize_planning_tools()
