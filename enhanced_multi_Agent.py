# Add memory context if specified
        memory_context = ""
        if memory_id and self.memory_manager:
            memories = self.memory_manager.get_memories(memory_id=memory_id)
            if memories:
                memory_context = "\n\nRelevant context from memory:\n"
                for memory in memories:
                    memory_content = memory["content"]
                    if isinstance(memory_content, dict):
                        memory_content = json.dumps(memory_content, indent=2)
                    memory_context += f"\n--- Memory from {memory['agent']} at {memory['timestamp']} ---\n{memory_content}\n"
        
        # Add conversation context if requested
        conversation_context_text = ""
        if conversation_context and memory_id and self.memory_manager:
            messages = self.memory_manager.get_conversation(memory_id)
            if messages:
                conversation_context_text = "\n\nPrevious conversation context:\n"
                for msg in messages:
                    role = msg.get("role", "unknown")
                    msg_content = msg.get("content", "")
                    conversation_context_text += f"\n{role}: {msg_content}\n"
                    
        # Add references to content if they exist
        enhanced_content = content + memory_context + conversation_context_text
        if references:
            enhanced_content += "\n\nReference information from other agents:\n"
            for agent, text in references.items():
                enhanced_content += f"\n--- {agent}'s output ---\n{text}\n"
        
        # Build the payload
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": enhanced_content
                }
            ]
        }
        
        # Add journey param if file is specified
        if file_param:
            with open(file_param, 'r', encoding='utf-8') as f:
                file_content = f.read()
            payload["journey"] = file_content
        
        # Escape single quotes for shell command
        payload_str = json.dumps(payload).replace("'", "'\\''")
        
        # Construct the curl command
        curl_command = f"""curl {CONFIG["endpoint"]} \\
  -H "Authorization: Bearer {CONFIG["api_key"]}" \\
  -H "Content-Type: application/json" \\
  -o "{self.output_file}" \\
  -d '{payload_str}'"""
        
        return curl_command
    
    def execute(self, content: str, file_param: Optional[str] = None, 
                read_from: List[str] = None, memory_id: Optional[str] = None,
                output_format: Optional[Dict[str, Any]] = None) -> Any:
        """Execute the API call and process structured output"""
        read_from = read_from or []
        cmd = self.create_api_call(content, file_param, read_from, memory_id)
        
        try:
            # Execute the command
            subprocess.run(cmd, shell=True, check=True)
            
            # Parse the output file to extract the actual response
            with open(self.output_file, 'r', encoding='utf-8') as f:
                raw_output = f.read()
            
            json_output = json.loads(raw_output)
            response_content = json_output.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            # Save the cleaned output
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write(response_content)
            
            # Process structured output if format is specified
            structured_output = response_content
            if output_format:
                try:
                    parser = OutputParser()
                    structured_output = parser.extract_format(response_content, output_format)
                except Exception as e:
                    print(f"Warning: Failed to parse structured output: {e}")
            
            # Store in memory if memory manager is available
            if self.memory_manager and memory_id:
                self.memory_manager.store_memory(
                    agent=self.name,
                    content=structured_output,
                    memory_id=memory_id,
                    metadata={"output_format": output_format}
                )
                
                # Update conversation context
                self.memory_manager.store_conversation(
                    messages=[{"role": "user", "content": content}, 
                              {"role": "assistant", "content": response_content}],
                    memory_id=memory_id
                )
            
            self.history.append({
                "input": content,
                "output": structured_output,
                "memory_id": memory_id
            })
            
            return structured_output
        except subprocess.CalledProcessError as e:
            print(f"Error with agent {self.name}:", e)
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing output for {self.name}:", e)
            raise
    
    def get_output(self) -> Optional[str]:
        """Get the output of this agent"""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None


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
            from_system = AgentSystem()  # Import here to avoid circular imports
            agent = from_system.get_agent(next_agent_name) or from_system.create_agent(
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


class AgentShell(cmd.Cmd):
    """Interactive CLI interface with enhanced features"""
    
    intro = "🤖 Enhanced Multi-Agent System Terminal\nType 'help' for available commands"
    prompt = "> "
    
    def __init__(self):
        super().__init__()
        self.system = AgentSystem()
    
    def do_create(self, arg):
        """Create a new agent: create <agent_name> [model] [type]"""
        args = arg.split()
        if len(args) < 1:
            print("Usage: create <agent_name> [model] [type]")
            return
        
        agent_name = args[0]
        model = args[1] if len(args) > 1 else CONFIG["default_model"]
        agent_type = args[2] if len(args) > 2 else "standard"
        
        self.system.create_agent(agent_name, model, agent_type)
        print(f"Agent {agent_name} created with model {model} and type {agent_type}")
    
    def do_run(self, arg):
        """Run an agent: run <agent_name> <prompt> [file] [ref:agent1,agent2] [memory:id] [format:json|markdown]"""
        parts = arg.split()
        if len(parts) < 2:
            print("Usage: run <agent_name> <prompt> [file] [ref:agent1,agent2] [memory:id] [format:json|markdown]")
            return
        
        agent_name = parts[0]
        # Get the agent or create if it doesn't exist
        agent = self.system.get_agent(agent_name) or self.system.create_agent(agent_name)
        
        # Extract file parameter if present
        file_param = None
        if "[file]" in arg:
            file_parts = arg.split("[file]")[1].split("[")[0].strip()
            file_param = file_parts
        
        # Extract references if present
        refs = []
        if "[ref:" in arg:
            ref_part = arg.split("[ref:")[1]
            if "]" in ref_part:
                ref_text = ref_part.split("]")[0]
                refs = [r.strip() for r in ref_text.split(",")]
        
        # Extract memory ID if present
        memory_id = None
        if "[memory:" in arg:
            memory_part = arg.split("[memory:")[1]
            if "]" in memory_part:
                memory_id = memory_part.split("]")[0].strip()
        
        # Extract output format if present
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
            for tag in ["file", "ref:", "memory:", "format:"]
        ])
        prompt = arg.replace(agent_name, "", 1).strip()[:prompt_end].strip()
        
        result = agent.execute(prompt, file_param, refs, memory_id, output_format)
        print(f"Result from {agent_name}:")
        if isinstance(result, dict):
            print(json.dumps(result, indent=2))
        else:
            print(result)
    
    def do_workflow(self, arg):
        """Execute a workflow from JSON file: workflow <workflow_file>"""
        args = arg.split()
        if len(args) < 1:
            print("Usage: workflow <workflow_file>")
            return
        
        workflow_file = args[0]
        if not os.path.exists(workflow_file):
            print(f"Workflow file {workflow_file} not found")
            return
        
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        print("Executing workflow...")
        results = self.system.execute_sequence(workflow)
        print("Workflow completed with results:")
        for agent, result in results.items():
            print(f"\n=== {agent} ===")
            if isinstance(result, dict):
                print(json.dumps(result, indent=2))
            else:
                print(result)
    
    def do_memory(self, arg):
        """Access memory: memory get <memory_id> [agent] or memory list [memory_id]"""
        args = arg.split()
        if len(args) < 1:
            print("Usage: memory get <memory_id> [agent] or memory list [memory_id]")
            return
        
        command = args[0]
        if command == "get" and len(args) >= 2:
            memory_id = args[1]
            agent = args[2] if len(args) > 2 else None
            
            memories = self.system.memory_manager.get_memories(memory_id, agent)
            if memories:
                print(f"Memories for {memory_id} {f'from {agent}' if agent else ''}:")
                for memory in memories:
                    print(f"\n--- {memory['agent']} at {memory['timestamp']} ---")
                    if isinstance(memory['content'], dict):
                        print(json.dumps(memory['content'], indent=2))
                    else:
                        print(memory['content'])
            else:
                print(f"No memories found for {memory_id} {f'from {agent}' if agent else ''}")
                
        elif command == "list":
            memory_id = args[1] if len(args) > 1 else None
            
            conn = sqlite3.connect(self.system.memory_manager.db_path)
            cursor = conn.cursor()
            
            if memory_id:
                cursor.execute("SELECT DISTINCT agent FROM memories WHERE memory_id = ?", (memory_id,))
                agents = [row[0] for row in cursor.fetchall()]
                
                if agents:
                    print(f"Agents with memories in {memory_id}:")
                    for agent in agents:
                        print(f"- {agent}")
                else:
                    print(f"No memories found for {memory_id}")
            else:
                cursor.execute("SELECT DISTINCT memory_id FROM memories")
                memory_ids = [row[0] for row in cursor.fetchall()]
                
                if memory_ids:
                    print("Available memory IDs:")
                    for mid in memory_ids:
                        print(f"- {mid}")
                else:
                    print("No memories stored")
            
            conn.close()
        else:
            print("Unknown memory command. Use 'memory get <memory_id>' or 'memory list'")
    
    def do_get(self, arg):
        """Get the output of an agent: get <agent_name>"""
        args = arg.split()
        if len(args) < 1:
            print("Usage: get <agent_name>")
            return
        
        agent_name = args[0]
        agent = self.system.get_agent(agent_name)
        if not agent:
            print(f"Agent {agent_name} not found")
            return
        
        output = agent.get_output()
        print(f"Output from {agent_name}:\n{output or '(no output)'}")
    
    def do_list(self, arg):
        """List all available agents"""
        print("Available agents:")
        for name, agent in self.system.agents.items():
            agent_type = "dynamic" if isinstance(agent, DynamicAgent) else "standard"
            print(f"- {name} ({agent.model}) [{agent_type}]")
    
    def do_exit(self, arg):
        """Exit the program"""
        print("Goodbye!")
        return True
    
    def default(self, line):
        print(f"Unknown command: {line}. Type 'help' for available commands.")


def main():
    """Main function"""
    # Check if we should run the CLI or execute a specific workflow
    if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
        if len(sys.argv) < 3:
            print("Please provide a workflow file path")
            sys.exit(1)
        
        workflow_file = sys.argv[2]
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        system = AgentSystem()
        try:
            results = system.execute_sequence(workflow)
            print("Workflow completed with results:")
            for agent, result in results.items():
                print(f"\n=== {agent} ===")
                if isinstance(result, dict):
                    print(json.dumps(result, indent=2))
                else:
                    print(result)
        except Exception as e:
            print("Workflow error:", e)
    else:
        # Start the interactive CLI
        AgentShell().cmdloop()


if __name__ == "__main__":
    main()!/usr/bin/env python3

import os
import json
import subprocess
import sys
import cmd
import uuid
import re
import sqlite3
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import jsonschema

# Configuration
CONFIG = {
    "output_dir": "./agent_outputs",
    "memory_dir": "./agent_memory",
    "default_model": "deepseek/deepseek-chat:free",
    "api_key": "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",
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "memory_db": "agent_memory.db"
}

# Ensure output directories exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["memory_dir"], exist_ok=True)


class MemoryManager:
    """Manages persistent memory across agent runs and sessions"""
    
    def __init__(self, db_path: str = CONFIG["memory_db"]):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            agent TEXT,
            timestamp TEXT,
            content TEXT,
            metadata TEXT
        )
        ''')
        
        # Create conversations table for storing conversation context
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            timestamp TEXT,
            messages TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_memory(self, agent: str, content: Any, memory_id: str = None, 
                     metadata: Dict[str, Any] = None) -> str:
        """Store a memory entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        memory_id = memory_id or "default"
        metadata = json.dumps(metadata or {})
        
        # Convert content to JSON string if it's not already a string
        if not isinstance(content, str):
            content = json.dumps(content)
        
        cursor.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, memory_id, agent, timestamp, content, metadata)
        )
        
        conn.commit()
        conn.close()
        
        return entry_id
    
    def get_memories(self, memory_id: str = None, agent: str = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve memories based on filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # Parse content and metadata from JSON
        for row in results:
            try:
                row["content"] = json.loads(row["content"])
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
            
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                row["metadata"] = {}
        
        conn.close()
        return results
    
    def store_conversation(self, messages: List[Dict[str, Any]], memory_id: str = None) -> str:
        """Store a conversation context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        memory_id = memory_id or "default"
        
        cursor.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?)",
            (conversation_id, memory_id, timestamp, json.dumps(messages))
        )
        
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def get_conversation(self, memory_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve the most recent conversation for a memory ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM conversations WHERE memory_id = ? ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query, (memory_id or "default",))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row["messages"])
        return []


class OutputParser:
    """Handles parsing and validation of structured outputs"""
    
    @staticmethod
    def parse_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text, even if it's embedded in other text"""
        # Try to find JSON pattern in the text
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*?})'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1) or match.group(2)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If no pattern match, try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON from text: {text[:100]}...")
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate JSON data against a schema"""
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, "Valid"
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
    
    @staticmethod
    def extract_format(text: str, output_format: Dict[str, Any]) -> Any:
        """Extract and validate output according to the specified format"""
        format_type = output_format.get("type", "text")
        
        if format_type == "json":
            data = OutputParser.parse_json(text)
            
            # Convert simple schema to jsonschema format
            if "schema" in output_format:
                schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for key, value_type in output_format["schema"].items():
                    if value_type == "string":
                        schema["properties"][key] = {"type": "string"}
                    elif value_type == "number":
                        schema["properties"][key] = {"type": "number"}
                    elif isinstance(value_type, list) and len(value_type) > 0:
                        schema["properties"][key] = {
                            "type": "array",
                            "items": {"type": value_type[0]}
                        }
                    schema["required"].append(key)
                
                valid, message = OutputParser.validate_schema(data, schema)
                if not valid:
                    print(f"Warning: Output does not match schema: {message}")
            
            return data
        
        elif format_type == "markdown":
            # For markdown, just return the text but we could add section extraction
            if "sections" in output_format:
                # Check if all required sections are present
                for section in output_format["sections"]:
                    if not re.search(rf"#{{{1,6}}}\s*{section}", text, re.IGNORECASE):
                        print(f"Warning: Missing section '{section}' in markdown output")
            
            return text
        
        # Default to returning the raw text
        return text


class Agent:
    """Agent class to handle API calls and maintain state"""
    
    def __init__(self, name: str, model: str = CONFIG["default_model"], 
                 memory_manager: Optional[MemoryManager] = None):
        self.name = name
        self.model = model
        self.output_file = os.path.join(CONFIG["output_dir"], f"{name}_output.txt")
        self.history = []
        self.memory_manager = memory_manager
    
    def create_api_call(self, content: str, file_param: Optional[str] = None, 
                        read_from: List[str] = None, memory_id: Optional[str] = None,
                        conversation_context: bool = False) -> str:
        """Create an API call with the given content and optional file"""
        read_from = read_from or []
        references = {}
        
        # If readFrom is specified, read content from other agents' outputs
        if read_from:
            for agent_name in read_from:
                # Special case for wildcard to read from all available agents
                if agent_name == "*":
                    for filename in os.listdir(CONFIG["output_dir"]):
                        if filename.endswith("_output.txt"):
                            other_agent = filename.replace("_output.txt", "")
                            if other_agent != self.name and other_agent not in read_from:
                                ref_file = os.path.join(CONFIG["output_dir"], filename)
                                with open(ref_file, 'r', encoding='utf-8') as f:
                                    references[other_agent] = f.read().strip()
                else:
                    ref_file = os.path.join(CONFIG["output_dir"], f"{agent_name}_output.txt")
                    if os.path.exists(ref_file):
                        with open(ref_file, 'r', encoding='utf-8') as f:
                            references[agent_name] = f.read().strip()
        
        # Add memory context if specified
        memory_context = ""
        if memory_id and self.memory_manager:
            memories = self.memory_manager.get_memories(memory_id=memory_id)
            if memories:
                memory_context = "\n\nRelevant context from memory:\n"
                for memory in memories:
                    memory_content = memory["content"]
                    if isinstance(memory_content, dict):
                        memory_content = json.dumps(memory_content, indent=2)
                    memory_context += f"\n--- Memory from {memory['agent']} at {memory['timestamp']} ---\n{memory_content}\n"
        
        # Add conversation context if requested
        conversation_context_text = ""
        if conversation_context and memory_id and self.memory_manager:
            messages = self.memory_manager.get_conversation(memory_id)
            if messages:
                conversation_context_text = "\n\nPrevious conversation context:\n"
                for msg in messages:
                    role = msg.get("role", "unknown")
                    msg_content = msg.get("content", "")
                    conversation_context_text += f"\n{role}: {msg_content}\n"
        
        #