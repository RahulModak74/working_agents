# agent_framework.py
# Core framework for the multi-agent system

import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Agent:
    """
    Base class for all agents in the system.
    """
    def __init__(self, 
                 agent_id: str, 
                 content: str,
                 tools: List[str] = None,
                 memory_id: str = None,
                 read_from: List[str] = None,
                 output_format: Dict[str, Any] = None):
        """
        Initialize a basic agent.
        
        Args:
            agent_id: Unique identifier for the agent
            content: The agent's instruction/content
            tools: List of tools the agent can use
            memory_id: ID of the memory this agent writes to
            read_from: List of agents this agent reads from
            output_format: Format specification for agent output
        """
        self.agent_id = agent_id
        self.content = content
        self.tools = tools or []
        self.memory_id = memory_id
        self.read_from = read_from or []
        self.output_format = output_format or {"type": "json"}
        
    def execute(self, inputs: Dict[str, Any] = None, memory_store: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the agent's task based on inputs and memory.
        
        Args:
            inputs: Input data for the agent
            memory_store: Access to the memory system
            
        Returns:
            Dict with the agent's output
        """
        logger.info(f"Executing agent: {self.agent_id}")
        
        try:
            # Prepare the context from inputs and memory
            context = self._prepare_context(inputs, memory_store)
            
            # Process the agent's content with the context
            processed_content = self._process_content(context)
            
            # Apply tools as specified
            result = self._apply_tools(processed_content, context)
            
            # Format the output according to the output_format specification
            formatted_output = self._format_output(result)
            
            return {
                "agent_id": self.agent_id,
                "status": "success",
                "output": formatted_output
            }
            
        except Exception as e:
            logger.error(f"Error executing agent {self.agent_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e)
            }
    
    def _prepare_context(self, inputs: Dict[str, Any], memory_store: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the context by gathering data from inputs and memory.
        
        Args:
            inputs: Input data
            memory_store: Memory system
            
        Returns:
            Dict with the prepared context
        """
        context = {}
        
        # Add all inputs to context
        if inputs:
            context.update(inputs)
        
        # Add data from agents we read from
        if self.read_from and memory_store:
            for source in self.read_from:
                if source == "*":
                    # Read all available data
                    for memory_id, memory_data in memory_store.items():
                        if memory_id not in context:
                            context[memory_id] = memory_data
                elif source in memory_store:
                    context[source] = memory_store[source]
        
        return context
    
    def _process_content(self, context: Dict[str, Any]) -> str:
        """
        Process the agent's content with the context.
        
        Args:
            context: The prepared context
            
        Returns:
            Processed content string
        """
        # Replace placeholders in content with context values
        processed_content = self.content
        
        # Simple placeholder replacement (for more complex, use a template engine)
        if '{' in processed_content and '}' in processed_content:
            try:
                # Find all placeholders like {key} and replace them
                import re
                placeholders = re.findall(r'\{([^}]+)\}', processed_content)
                for placeholder in placeholders:
                    if placeholder in context:
                        processed_content = processed_content.replace(
                            f"{{{placeholder}}}", str(context[placeholder])
                        )
            except Exception as e:
                logger.warning(f"Error processing content placeholders: {str(e)}")
        
        return processed_content
    
    def _apply_tools(self, processed_content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the agent's tools to the processed content.
        
        Args:
            processed_content: The processed content
            context: The context data
            
        Returns:
            Dict with the results of applying tools
        """
        from agent_tools import ToolRegistry
        
        result = {"raw_content": processed_content}
        
        # Apply each tool in sequence
        for tool_name in self.tools:
            try:
                # Get the tool function from the registry
                tool_function = ToolRegistry.get_tool(tool_name)
                
                if tool_function:
                    # Apply the tool and add its output to the result
                    tool_output = tool_function(processed_content, context)
                    
                    # Extract the operation name from the tool name
                    operation = tool_name.split(":")[-1] if ":" in tool_name else tool_name
                    result[f"{operation}_output"] = tool_output
                    
                    # The last tool's output becomes the main result
                    result["main_output"] = tool_output
                else:
                    logger.warning(f"Tool not found: {tool_name}")
            except Exception as e:
                logger.error(f"Error applying tool {tool_name}: {str(e)}")
                logger.error(traceback.format_exc())
                result[f"{tool_name}_error"] = str(e)
        
        # If no tools were applied or none produced a main output, use the raw content
        if "main_output" not in result:
            result["main_output"] = processed_content
            
        return result
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the output according to the output_format specification.
        
        Args:
            result: The raw result to format
            
        Returns:
            Formatted output
        """
        # Get the main result
        main_output = result.get("main_output", {})
        
        # Initialize formatted output
        formatted_output = {}
        
        if self.output_format["type"] == "json":
            # Format as JSON
            if "schema" in self.output_format:
                # Try to conform to the schema
                schema = self.output_format["schema"]
                formatted_output = self._conform_to_schema(main_output, schema)
            else:
                # Just use the main output
                formatted_output = main_output
                
        elif self.output_format["type"] == "markdown":
            # Format as Markdown
            if "sections" in self.output_format:
                # Create a document with the specified sections
                sections = self.output_format["sections"]
                formatted_output = {"content": ""}
                
                for section in sections:
                    section_content = main_output.get(section, f"Content for {section}")
                    formatted_output["content"] += f"## {section}\n\n{section_content}\n\n"
                    
                    # Also include each section separately
                    formatted_output[section] = section_content
            else:
                # Just convert the main output to Markdown
                formatted_output = {"content": str(main_output)}
        
        return formatted_output
    
    def _conform_to_schema(self, data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Try to conform data to a specified schema.
        
        Args:
            data: The data to conform
            schema: The schema to conform to
            
        Returns:
            Data conforming to the schema
        """
        # This is a simplified implementation
        # A real implementation would validate and transform data more thoroughly
        
        if isinstance(schema, dict):
            result = {}
            for key, value_schema in schema.items():
                if isinstance(value_schema, dict):
                    # Recursive call for nested objects
                    nested_data = data.get(key, {}) if isinstance(data, dict) else {}
                    result[key] = self._conform_to_schema(nested_data, value_schema)
                elif isinstance(value_schema, list) and value_schema:
                    # Handle list schemas
                    if isinstance(data, dict) and key in data and isinstance(data[key], list):
                        # If data has the key and it's a list, use it
                        result[key] = data[key]
                    else:
                        # Create a dummy list with one item conforming to the schema
                        if len(value_schema) > 0 and isinstance(value_schema[0], dict):
                            # If the list schema has a dict item, create a dummy item
                            result[key] = [self._conform_to_schema({}, value_schema[0])]
                        else:
                            # Otherwise, just use an empty string as the dummy item
                            result[key] = [""]
                elif value_schema == "string":
                    if isinstance(data, dict) and key in data:
                        result[key] = str(data[key])
                    else:
                        result[key] = ""
                elif value_schema == "number":
                    if isinstance(data, dict) and key in data:
                        try:
                            result[key] = float(data[key])
                        except (ValueError, TypeError):
                            result[key] = 0.0
                    else:
                        result[key] = 0.0
                elif value_schema == "boolean":
                    if isinstance(data, dict) and key in data:
                        result[key] = bool(data[key])
                    else:
                        result[key] = False
                else:
                    if isinstance(data, dict) and key in data:
                        result[key] = data[key]
                    else:
                        result[key] = None
            return result
        
        # If schema is not a dict, return data as is or a default value
        return data if data is not None else ""


class DynamicAgent(Agent):
    """
    Agent that can dynamically select from multiple actions.
    """
    def __init__(self,
                 agent_id: str,
                 content: str,
                 tools: List[str] = None,
                 memory_id: str = None,
                 read_from: List[str] = None,
                 output_format: Dict[str, Any] = None,
                 initial_prompt: str = None,
                 actions: Dict[str, Dict[str, Any]] = None):
        """
        Initialize a dynamic agent that can select from multiple actions.
        
        Args:
            agent_id: Unique identifier for the agent
            content: The agent's instruction/content
            tools: List of tools the agent can use
            memory_id: ID of the memory this agent writes to
            read_from: List of agents this agent reads from
            output_format: Format specification for agent output
            initial_prompt: Prompt to determine which action to take
            actions: Dict of possible actions the agent can take
        """
        super().__init__(agent_id, content, tools, memory_id, read_from, output_format)
        self.initial_prompt = initial_prompt
        self.actions = actions or {}
        
    def execute(self, inputs: Dict[str, Any] = None, memory_store: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute the dynamic agent, selecting and executing an action.
        
        Args:
            inputs: Input data for the agent
            memory_store: Access to the memory system
            
        Returns:
            Dict with the agent's output and selected action
        """
        logger.info(f"Executing dynamic agent: {self.agent_id}")
        
        try:
            # Prepare the context
            context = self._prepare_context(inputs, memory_store)
            
            # Process the initial prompt
            processed_prompt = self._process_content(context) if self.initial_prompt else ""
            
            # Determine which action to take
            selected_action, action_rationale = self._select_action(context, processed_prompt)
            
            # If no action was selected, return an error
            if not selected_action:
                return {
                    "agent_id": self.agent_id,
                    "status": "error",
                    "error": "No action selected"
                }
            
            # Get the selected action configuration
            action_config = self.actions.get(selected_action)
            
            # Return the action selection result
            result = {
                "agent_id": self.agent_id,
                "status": "success",
                "selected_action": selected_action,
                "action_rationale": action_rationale,
                "action_agent": action_config.get("agent")
            }
            
            # Format output according to the specification
            formatted_output = self._format_output({
                "main_output": {
                    "selected_approach": selected_action,
                    "justification": action_rationale,
                    "expected_benefits": ["Tailored approach for this problem type"]
                }
            })
            
            result["output"] = formatted_output
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing dynamic agent {self.agent_id}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "agent_id": self.agent_id,
                "status": "error",
                "error": str(e)
            }
    
    def _select_action(self, context: Dict[str, Any], processed_prompt: str) -> Tuple[str, str]:
        """
        Select which action to take based on the context and processed prompt.
        
        Args:
            context: The context data
            processed_prompt: The processed initial prompt
            
        Returns:
            Tuple of (selected_action, rationale)
        """
        logger.info(f"Selecting action for agent: {self.agent_id}")
        
        # Get available actions
        available_actions = list(self.actions.keys())
        
        if not available_actions:
            return None, "No actions available"
        
        # In a real implementation, this would use LLM or other decision-making process
        # For now, we'll use simple keyword matching
        
        # Convert context and prompt to a text for analysis
        analysis_text = str(context) + " " + processed_prompt
        
        # Simple keyword matching (in a real system, use more sophisticated methods)
        matched_actions = []
        for action in available_actions:
            # Check if action name or related keywords appear in the text
            action_keywords = [action] + self._get_action_keywords(action)
            if any(keyword.lower() in analysis_text.lower() for keyword in action_keywords):
                matched_actions.append(action)
        
        if matched_actions:
            # Choose the best match
            selected_action = matched_actions[0]
            rationale = f"Selected '{selected_action}' because it matches the context and requirements"
        else:
            # Fallback to the first action
            selected_action = available_actions[0]
            rationale = f"Selected '{selected_action}' as the default approach"
        
        return selected_action, rationale
    
    def _get_action_keywords(self, action: str) -> List[str]:
        """
        Get keywords related to an action for matching.
        
        Args:
            action: The action name
            
        Returns:
            List of related keywords
        """
        # Define related keywords for each action
        action_keywords = {
            "tree_of_thoughts": ["tree", "branching", "exploration", "paths", "multiple solutions"],
            "multi_agent_debate": ["debate", "discussion", "argue", "perspective", "positions"],
            "metacognitive_reflection": ["metacognition", "reflection", "bias", "thinking about thinking"]
        }
        
        return action_keywords.get(action, [])


class AgentSystem:
    """
    System for managing and executing a collection of agents.
    """
    def __init__(self, name: str):
        """
        Initialize an agent system.
        
        Args:
            name: The name of the agent system
        """
        self.name = name
        self.agents = {}  # Dictionary of agent_id to Agent objects
        self.memory_store = {}  # Memory storage for the system
        
    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the system.
        
        Args:
            agent: The agent to add
        """
        self.agents[agent.agent_id] = agent
        logger.info(f"Added agent '{agent.agent_id}' to system '{self.name}'")
        
    def add_agents_from_config(self, config: List[Dict[str, Any]]) -> None:
        """
        Add multiple agents from a configuration list.
        
        Args:
            config: List of agent configurations
        """
        for agent_config in config:
            agent_id = agent_config.get("agent")
            
            # Skip if no agent ID
            if not agent_id:
                logger.warning("Skipping agent without ID")
                continue
            
            # Determine if this is a dynamic agent
            is_dynamic = agent_config.get("type") == "dynamic"
            
            # Create the appropriate agent type
            if is_dynamic:
                # Process actions for dynamic agents
                actions = {}
                if "actions" in agent_config:
                    for action_name, action_config in agent_config["actions"].items():
                        actions[action_name] = {
                            "agent": action_config.get("agent"),
                            "content": action_config.get("content"),
                            "readFrom": action_config.get("readFrom", []),
                            "tools": action_config.get("tools", [])
                        }
                
                # Create the dynamic agent
                agent = DynamicAgent(
                    agent_id=agent_id,
                    content=agent_config.get("content", ""),
                    tools=agent_config.get("tools", []),
                    memory_id=agent_config.get("memory_id"),
                    read_from=agent_config.get("readFrom", []),
                    output_format=agent_config.get("output_format"),
                    initial_prompt=agent_config.get("initial_prompt"),
                    actions=actions
                )
            else:
                # Create a regular agent
                agent = Agent(
                    agent_id=agent_id,
                    content=agent_config.get("content", ""),
                    tools=agent_config.get("tools", []),
                    memory_id=agent_config.get("memory_id"),
                    read_from=agent_config.get("readFrom", []),
                    output_format=agent_config.get("output_format")
                )
            
            # Add the agent to the system
            self.add_agent(agent)
        
        logger.info(f"Added {len(config)} agents to system '{self.name}'")
    
    def execute_workflow(self, entry_point: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a workflow starting with the specified entry point agent.
        
        Args:
            entry_point: The ID of the agent to start with
            inputs: Input data for the workflow
            
        Returns:
            Dict with the workflow results
        """
        logger.info(f"Executing workflow starting with agent '{entry_point}'")
        
        if inputs is None:
            inputs = {}
        
        # Initialize the results
        results = {}
        
        # Set up the agent queue
        agent_queue = [entry_point]
        visited = set()
        
        # Process agents until the queue is empty
        while agent_queue:
            current_agent_id = agent_queue.pop(0)
            
            # Skip if already visited
            if current_agent_id in visited:
                continue
            
            # Skip if the agent doesn't exist
            if current_agent_id not in self.agents:
                logger.error(f"Agent '{current_agent_id}' not found")
                continue
            
            # Execute the current agent
            current_agent = self.agents[current_agent_id]
            result = current_agent.execute(inputs, self.memory_store)
            
            # Store the result
            results[current_agent_id] = result
            
            # Store in memory if specified
            if current_agent.memory_id:
                if current_agent.memory_id not in self.memory_store:
                    self.memory_store[current_agent.memory_id] = {}
                self.memory_store[current_agent.memory_id][current_agent_id] = result
            
            # Mark as visited
            visited.add(current_agent_id)
            
            # For dynamic agents, add the selected action's agent to the queue
            if isinstance(current_agent, DynamicAgent) and "selected_action" in result:
                selected_action = result["selected_action"]
                if selected_action in current_agent.actions:
                    action_agent = current_agent.actions[selected_action].get("agent")
                    if action_agent and action_agent not in visited and action_agent not in agent_queue:
                        agent_queue.append(action_agent)
            
            # Add agents that depend on this one to the queue
            for agent_id, agent in self.agents.items():
                if agent_id not in visited and agent_id not in agent_queue:
                    # Check if this agent reads from the current agent
                    reads_from_current = False
                    for source in agent.read_from:
                        if source == "*" or source == current_agent_id:
                            reads_from_current = True
                            break
                    
                    if reads_from_current:
                        agent_queue.append(agent_id)
        
        logger.info(f"Workflow completed with {len(results)} agents executed")
        return results
