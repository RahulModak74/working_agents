#!/usr/bin/env python3

from typing import List, Dict, Any, Optional

from config import CONFIG
from memory_manager import MemoryManager
from agent import Agent
from dynamic_agent import DynamicAgent


class AgentSystem:
    """Agent System class to manage multiple agents"""
    
    def __init__(self):
        self.agents = {}
        self.memory_manager = MemoryManager()
    
    def create_agent(self, name: str, model: str = CONFIG["default_model"],
                     agent_type: str = "standard") -> Agent:
        """Create a new agent of the specified type"""
        if agent_type == "dynamic":
            self.agents[name] = DynamicAgent(name, model, self.memory_manager)
        else:
            self.agents[name] = Agent(name, model, self.memory_manager)
        return self.agents[name]
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def execute_sequence(self, sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a sequence of agent calls with enhanced features"""
        results = {}
        
        for step in sequence:
            agent_name = step["agent"]
            agent_type = step.get("type", "standard")
            
            # Create agent if it doesn't exist
            if agent_name not in self.agents:
                self.create_agent(agent_name, CONFIG.get("default_model"), agent_type)
            
            agent = self.agents[agent_name]
            
            # Handle dynamic agent differently
            if agent_type == "dynamic" and isinstance(agent, DynamicAgent):
                print(f"🤖 Running dynamic agent: {agent_name}")
                action_key, result = agent.execute_dynamic(
                    initial_prompt=step.get("initial_prompt", ""),
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
                # Standard agent execution
                print(f"🤖 Running agent: {agent_name}")
                result = agent.execute(
                    content=step.get("content", ""),
                    file_param=step.get("file"),
                    read_from=step.get("readFrom", []),
                    memory_id=step.get("memory_id"),
                    output_format=step.get("output_format")
                )
                results[agent_name] = result
                print(f"✅ {agent_name} completed")
        
        return results