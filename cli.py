#!/usr/bin/env python3

import json
import cmd
import os
import sqlite3
from typing import List, Dict, Any

from config import CONFIG
from agent_system import AgentSystem


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
            from dynamic_agent import DynamicAgent
            agent_type = "dynamic" if isinstance(agent, DynamicAgent) else "standard"
            print(f"- {name} ({agent.model}) [{agent_type}]")
    
    def do_exit(self, arg):
        """Exit the program"""
        print("Goodbye!")
        return True
    
    def default(self, line):
        print(f"Unknown command: {line}. Type 'help' for available commands.")