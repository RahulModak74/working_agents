#!/usr/bin/env python3

import sys
import os
import json

def modify_workflow(workflow_file, output_file=None):
    """
    Modify the workflow to reduce timeouts in the source_collector agent
    """
    # Load the workflow file
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)
    
    # Find the source_collector agent
    for i, agent in enumerate(workflow):
        if agent.get('agent') == 'source_collector':
            # Modify the source_collector agent to limit the scope
            source_collector = agent.copy()
            source_collector['content'] = "Collect and validate only the most important sources used in researching 'Cyber security evolving role with AI'. Limit to 5-10 key sources per dimension and ensure proper citation format."
            # Set a shorter timeout using the OpenRouter API parameter
            source_collector['timeout'] = 30  # 30 seconds timeout
            # Update the agent in the workflow
            workflow[i] = source_collector
            break
    
    # Find the comprehensive_report_generator agent
    for i, agent in enumerate(workflow):
        if agent.get('agent') == 'comprehensive_report_generator':
            # Modify to work without relying on the source_collector
            report_generator = agent.copy()
            # Add a reference to tools that are available
            report_generator['tools'] = ["research:cite_sources", "memory:retrieve_all"]
            # Update the content to handle missing tools gracefully
            report_generator['content'] = "Generate a comprehensive research report on 'Cyber security evolving role with AI' with citations to key sources. Include analysis of all dimensions and cross-dimensional insights, focusing on quality over quantity of citations."
            workflow[i] = report_generator
            break
    
    # Save the modified workflow
    if output_file is None:
        output_file = workflow_file.replace('.json', '_modified.json')
    
    with open(output_file, 'w') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"Modified workflow saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_workflow.py <workflow_file> [output_file]")
        sys.exit(1)
    
    workflow_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    modify_workflow(workflow_file, output_file)
