#!/usr/bin/env python3

import sys
import os
import json
import re
from typing import Dict, Any, List, Optional

# Import specialized runners 
from cyber_workflow_runner import run_cyber_security_workflow
from journey_workflow_runner import run_customer_journey_analysis
from agent_system import AgentSystem
from cli import AgentShell

def identify_workflow_type(workflow_file: str) -> str:
    """
    Identify the type of workflow based on the workflow file content
    Returns: 'cyber', 'journey', or 'generic'
    """
    try:
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # Check for cyber security workflow indicators
        cyber_keywords = ['security', 'threat', 'attack', 'vulnerability', 'suspicious', 'block_traffic']
        
        # Check for customer journey workflow indicators
        journey_keywords = ['customer', 'journey', 'segment', 'conversion', 'roi', 'path', 'segment_targeting']
        
        # Flatten the workflow to a string for keyword searching
        workflow_str = json.dumps(workflow).lower()
        
        # Count matches
        cyber_matches = sum(1 for keyword in cyber_keywords if keyword in workflow_str)
        journey_matches = sum(1 for keyword in journey_keywords if keyword in workflow_str)
        
        # Determine the type based on the highest match count
        if cyber_matches > journey_matches:
            return 'cyber'
        elif journey_matches > cyber_matches:
            return 'journey'
        
        # If the file contains 'cyber' in its name, classify as cyber
        if 'cyber' in workflow_file.lower():
            return 'cyber'
        
        # If the file contains 'journey' in its name, classify as journey
        if 'journey' in workflow_file.lower():
            return 'journey'
        
        # If we can't determine, use a generic approach
        return 'generic'
    
    except Exception as e:
        print(f"Error identifying workflow type: {e}")
        return 'generic'

def main():
    """Main function"""
    # Check if we should run the CLI or execute a specific workflow
    if len(sys.argv) > 1 and sys.argv[1] == "--workflow":
        if len(sys.argv) < 3:
            print("Usage: python main.py --workflow <workflow_file> [csv_file]")
            sys.exit(1)
        
        workflow_file = sys.argv[2]
        
        # Check if workflow file exists
        if not os.path.exists(workflow_file):
            print(f"Workflow file not found: {workflow_file}")
            sys.exit(1)
        
        # Determine if a CSV file is provided
        csv_file = None
        if len(sys.argv) > 3:
            csv_file = sys.argv[3]
            if not os.path.exists(csv_file):
                print(f"CSV file not found: {csv_file}")
                sys.exit(1)
        
        # Identify workflow type
        workflow_type = identify_workflow_type(workflow_file)
        
        # Load the workflow
        with open(workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        # Execute workflow based on type
        if workflow_type == 'cyber':
            if not csv_file:
                print("CSV file is required for cyber security workflow")
                sys.exit(1)
            
            print(f"Executing cyber security workflow: {workflow_file}")
            results = run_cyber_security_workflow(workflow_file, csv_file)
            
            print("\nWorkflow completed with results:")
            for agent, result in results.items():
                if "_action" in agent:
                    print(f"\n=== {agent} ===")
                    print(result)
                elif isinstance(result, dict) and "error" in result:
                    print(f"\n=== {agent} ===")
                    print(f"Error: {result['error']}")
                else:
                    print(f"\n=== {agent} ===")
                    if isinstance(result, dict):
                        print("✓ Success")
                    else:
                        print(result)
                
        elif workflow_type == 'journey':
            if not csv_file:
                print("CSV file is required for customer journey workflow")
                sys.exit(1)
                
            print(f"Executing customer journey workflow: {workflow_file}")
            results = run_customer_journey_analysis(workflow_file, csv_file)
            
            print("\nAnalysis Summary:")
            for agent, result in results.items():
                print(f"- {agent}: {'Success' if 'error' not in result else 'Error: ' + result['error']}")
                
        else:
            # Fall back to the generic agent system
            print(f"Executing generic workflow: {workflow_file}")
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
