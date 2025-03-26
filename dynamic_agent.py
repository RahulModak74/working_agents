#!/usr/bin/env python3

from typing import List, Dict, Any, Optional, Tuple

from config import CONFIG
from agent import Agent
from memory_manager import MemoryManager


class DynamicAgent(Agent):
    """Agent that can choose its next action dynamically"""
    
    def __init__(self, name: str, model: str = CONFIG["default_model"], 
                 memory_manager: Optional[MemoryManager] = None):
        super().__init__(name, model, memory_manager)
    
    def execute_dynamic(self, initial_prompt: str, file_param: Optional[str] = None,
                        read_from: List[str] = None, memory_id: Optional[str] = None,
                        output_format: Optional[Dict[str, Any]] = None,
                        actions: Dict[str, Dict[str, Any]] = None) -> Tuple[str, Any]:
        """Execute the initial prompt and then determine the next action"""
        # First, run the initial prompt to get the decision
        decision = self.execute(
            initial_prompt, file_param, read_from, memory_id, output_format
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
            
            # Execute the next agent with the specified parameters
            next_result = agent.execute(
                action.get("content", ""),
                action.get("file", None),
                action.get("readFrom", read_from),
                memory_id,
                action.get("output_format", None)
            )
            
            return action_key, next_result
        
        return action_key, decision