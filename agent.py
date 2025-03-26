#!/usr/bin/env python3

import os
import json
import subprocess
from typing import List, Dict, Any, Optional

from config import CONFIG
from memory_manager import MemoryManager
from output_parser import OutputParser


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