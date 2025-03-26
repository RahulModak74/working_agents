#!/usr/bin/env python3

import sys
import json

from agent_system import AgentSystem
from cli import AgentShell


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
    main()