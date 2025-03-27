#!/usr/bin/env python3

import sys
import json
import os

from agent_system import AgentSystem
from cli import AgentShell

# Check if tools module is available and import it
try:
    # Add COMPONENT directory to path if it exists
    component_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COMPONENT")
    if os.path.exists(component_dir):
        sys.path.append(component_dir)
    
    # Try to import tool-related modules
    from tools import TOOL_MANAGER
    tools_available = True
except ImportError:
    tools_available = False


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
    
    # Handle tool workflow execution
    elif len(sys.argv) > 1 and sys.argv[1] == "--tool-workflow":
        if not tools_available:
            print("Error: Tools module not available. Please install the required dependencies:")
            print("pip install -r COMPONENT/requirement_tools.txt")
            sys.exit(1)
            
        if len(sys.argv) < 3:
            print("Please provide a workflow file path")
            sys.exit(1)
        
        workflow_file = sys.argv[2]
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        system = AgentSystem()
        try:
            # Import execute_sequence_with_tools dynamically
            if hasattr(system, 'execute_sequence_with_tools'):
                results = system.execute_sequence_with_tools(workflow)
            else:
                # Fall back to regular execution if the tools integration isn't complete
                print("Warning: execute_sequence_with_tools not found, falling back to regular execution")
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
            import traceback
            traceback.print_exc()
    else:
        # Start the interactive CLI
        AgentShell().cmdloop()


if __name__ == "__main__":
    main()
