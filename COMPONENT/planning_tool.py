#!/usr/bin/env python3

import json
import re
from typing import Dict, Any, Optional, Union, Tuple, List

class PlanningTool:
    """Tool for implementing various planning techniques (CoT, ReAct)"""
    
    def __init__(self, max_steps: int = 10, verbose: bool = False):
        """Initialize the planning tool with configuration options"""
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []
        self.current_plan = None
    
    def chain_of_thought(self, problem: str, steps: List[str] = None) -> Dict[str, Any]:
        """Implement chain-of-thought reasoning
        
        Args:
            problem: The problem statement or question to solve
            steps: Optional list of reasoning steps to follow
            
        Returns:
            Dict containing the reasoning chain and final answer
        """
        reasoning_chain = []
        
        # If specific steps are provided, use them as the reasoning template
        if steps:
            reasoning_template = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
            reasoning_prompt = f"Problem: {problem}\n\nReasoning steps:\n{reasoning_template}"
        else:
            # Otherwise, use a general CoT prompt
            reasoning_prompt = (
                f"Problem: {problem}\n\n"
                f"Let's think through this step-by-step:\n"
                f"1. "
            )
        
        # This would typically go to an LLM, but we're keeping this implementation-agnostic
        # The agent implementation will handle the actual prompt and response
        
        result = {
            "problem": problem,
            "reasoning_prompt": reasoning_prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": None
        }
        
        self.history.append({
            "type": "chain_of_thought",
            "problem": problem,
            "steps": steps,
            "result": result
        })
        
        return result
    
    def react(self, task: str, tools: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Implement ReAct (Reasoning + Acting) approach
        
        Args:
            task: The task description
            tools: List of available tools and their descriptions
            
        Returns:
            Dict containing the reasoning, actions and final result
        """
        # Initialize the ReAct process
        react_cycles = []
        
        # Create tool descriptions for the prompt
        tool_descriptions = ""
        if tools:
            tool_descriptions = "Available tools:\n"
            for tool in tools:
                tool_descriptions += f"- {tool['name']}: {tool['description']}\n"
        
        # Create the initial ReAct prompt
        react_prompt = (
            f"Task: {task}\n\n"
            f"{tool_descriptions}\n"
            f"Let's solve this step by step:\n\n"
            f"Thought 1: "
        )
        
        result = {
            "task": task,
            "react_prompt": react_prompt,
            "cycles": react_cycles,
            "final_result": None
        }
        
        self.history.append({
            "type": "react",
            "task": task,
            "tools": tools,
            "result": result
        })
        
        return result
    
    def extract_reasoning_and_actions(self, text: str) -> Dict[str, Any]:
        """Extract reasoning, actions and observations from a ReAct response
        
        Args:
            text: Raw text from an agent's ReAct response
            
        Returns:
            Dict containing parsed thoughts, actions and observations
        """
        # Extract Thoughts
        thoughts = []
        thought_pattern = r"Thought\s+\d+:\s*(.*?)(?=\n*(?:Action\s+\d+:|Observation\s+\d+:|Thought\s+\d+:|$))"
        for match in re.finditer(thought_pattern, text, re.DOTALL):
            thoughts.append(match.group(1).strip())
        
        # Extract Actions
        actions = []
        action_pattern = r"Action\s+\d+:\s*(.*?)(?=\n*(?:Observation\s+\d+:|Thought\s+\d+:|$))"
        for match in re.finditer(action_pattern, text, re.DOTALL):
            actions.append(match.group(1).strip())
        
        # Extract Observations
        observations = []
        observation_pattern = r"Observation\s+\d+:\s*(.*?)(?=\n*(?:Action\s+\d+:|Thought\s+\d+:|$))"
        for match in re.finditer(observation_pattern, text, re.DOTALL):
            observations.append(match.group(1).strip())
        
        # Extract Final Answer if exists
        final_answer = None
        final_answer_pattern = r"Final Answer:\s*(.*?)$"
        match = re.search(final_answer_pattern, text, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
        
        return {
            "thoughts": thoughts,
            "actions": actions,
            "observations": observations,
            "final_answer": final_answer
        }
    
    def create_plan(self, goal: str, subtasks: List[str] = None) -> Dict[str, Any]:
        """Create a structured plan with tasks and subtasks
        
        Args:
            goal: The overall goal of the plan
            subtasks: Optional list of subtasks
            
        Returns:
            Dict containing the planning structure
        """
        plan = {
            "goal": goal,
            "subtasks": subtasks or [],
            "status": "created",
            "current_subtask": 0,
            "results": []
        }
        
        self.current_plan = plan
        
        self.history.append({
            "type": "create_plan",
            "goal": goal,
            "subtasks": subtasks,
            "plan": plan
        })
        
        return plan
    
    def update_plan(self, plan_update: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing plan
        
        Args:
            plan_update: Dict containing updates to the plan
            
        Returns:
            Dict containing the updated plan
        """
        if not self.current_plan:
            return {"error": "No active plan to update"}
        
        # Update the current plan with new information
        for key, value in plan_update.items():
            if key in self.current_plan:
                self.current_plan[key] = value
        
        self.history.append({
            "type": "update_plan",
            "plan_update": plan_update,
            "updated_plan": self.current_plan
        })
        
        return self.current_plan
    
    def execute_subtask(self, subtask_index: int, result: Any = None) -> Dict[str, Any]:
        """Execute a subtask in the current plan
        
        Args:
            subtask_index: Index of the subtask to execute
            result: Result of the subtask execution
            
        Returns:
            Dict containing the updated plan
        """
        if not self.current_plan:
            return {"error": "No active plan"}
        
        if subtask_index >= len(self.current_plan["subtasks"]):
            return {"error": "Subtask index out of range"}
        
        # Update the current subtask
        self.current_plan["current_subtask"] = subtask_index
        
        # Add the result
        if len(self.current_plan["results"]) <= subtask_index:
            # Extend the results list if needed
            self.current_plan["results"].extend([None] * (subtask_index + 1 - len(self.current_plan["results"])))
        
        self.current_plan["results"][subtask_index] = result
        
        # Check if all subtasks are completed
        if all(r is not None for r in self.current_plan["results"]) and len(self.current_plan["results"]) == len(self.current_plan["subtasks"]):
            self.current_plan["status"] = "completed"
        else:
            self.current_plan["status"] = "in_progress"
        
        self.history.append({
            "type": "execute_subtask",
            "subtask_index": subtask_index,
            "result": result,
            "updated_plan": self.current_plan
        })
        
        return self.current_plan
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the current plan
        
        Returns:
            Dict containing a summary of the plan
        """
        if not self.current_plan:
            return {"error": "No active plan"}
        
        completed_tasks = sum(1 for r in self.current_plan["results"] if r is not None)
        total_tasks = len(self.current_plan["subtasks"])
        
        return {
            "goal": self.current_plan["goal"],
            "status": self.current_plan["status"],
            "progress": f"{completed_tasks}/{total_tasks}",
            "completion_percentage": int(completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "current_subtask": self.current_plan["subtasks"][self.current_plan["current_subtask"]] if self.current_plan["subtasks"] else None
        }
    
    def parse_agent_plan(self, plan_text: str) -> Dict[str, Any]:
        """Parse a plan created by an agent in free text format
        
        Args:
            plan_text: The plan text to parse
            
        Returns:
            Dict containing the structured plan
        """
        # Extract the goal
        goal = None
        goal_pattern = r"(?:Goal|Objective):\s*(.*?)(?=\n|$)"
        match = re.search(goal_pattern, plan_text, re.IGNORECASE)
        if match:
            goal = match.group(1).strip()
        
        # Extract subtasks
        subtasks = []
        subtask_pattern = r"(?:\d+\.\s*(.*?)(?=\n\d+\.|\n*$))"
        for match in re.finditer(subtask_pattern, plan_text, re.DOTALL):
            subtask = match.group(1).strip()
            if subtask:
                subtasks.append(subtask)
        
        # Create a plan from the parsed information
        if goal:
            return self.create_plan(goal, subtasks)
        else:
            return {"error": "Could not parse plan - no goal found"}


def get_planning_tool(max_steps: int = 10, verbose: bool = False) -> PlanningTool:
    """Factory function to get a planning tool instance"""
    return PlanningTool(max_steps, verbose)
