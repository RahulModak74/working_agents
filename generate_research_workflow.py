#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
import subprocess
import re
from typing import Dict, Any, List

def clean_topic_for_filename(topic: str) -> str:
    """Convert a topic string into a clean filename string."""
    cleaned = re.sub(r'[^\w\s-]', '', topic).strip().lower()
    return re.sub(r'[-\s]+', '_', cleaned)

def parse_user_input(input_file: str) -> Dict[str, Any]:
    """Parse the user input file and extract research parameters."""
    research_request = {}
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    
    # Parse based on file type
    if input_file.endswith('.csv'):
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # If only one line, assume it's just the research topic
            if len(lines) == 1:
                research_request['topic'] = lines[0].strip()
                research_request['dimensions'] = []
            
            # Multiple lines - header + values format
            elif len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(',')]
                values = [v.strip() for v in lines[1].split(',')]
                
                data = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
                research_request['topic'] = data.get('topic', '')
                
                # Extract dimensions
                dimensions = []
                for i in range(1, 10):
                    dim_key = f'dimension{i}'
                    if dim_key in data and data[dim_key]:
                        dimensions.append(data[dim_key])
                
                research_request['dimensions'] = dimensions
    
    elif input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            research_request['topic'] = data.get('topic', '')
            research_request['dimensions'] = data.get('dimensions', [])
    
    else:
        # Plain text
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            lines = content.split('\n')
            research_request['topic'] = lines[0] if lines else content
            research_request['dimensions'] = []
    
    return research_request

def validate_input(research_request: Dict[str, Any]) -> tuple:
    """Validate the input and return (is_valid, message)."""
    if not research_request.get('topic'):
        return False, "Error: No research topic provided."
    
    if len(research_request.get('topic', '').split()) <= 1:
        return False, f"Error: The topic '{research_request.get('topic')}' is too vague."
    
    return True, "Input validation passed."

def load_workflow_template() -> List[Dict[str, Any]]:
    """Load the workflow template from file or use default embedded template."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "congnitive_optimized_memory_research_workflow.json")
    
    # Try all possible template locations
    if not os.path.exists(template_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alternate_paths = [
            os.path.join(script_dir, "cognitive_optimized_memory_research_workflow.json"),
            os.path.join(script_dir, "cognitive_optimized_memory_research_workflow_template.json"),
            os.path.join(script_dir, "templates", "cognitive_optimized_memory_research_workflow.json"),
        ]
        
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                template_path = alt_path
                break
    
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("Warning: No template file found. Using embedded template.")
        return COMPREHENSIVE_WORKFLOW_TEMPLATE

def customize_workflow(workflow: List[Dict[str, Any]], research_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Customize the workflow with the research parameters."""
    custom_workflow = json.loads(json.dumps(workflow))
    
    research_topic = research_request['topic']
    dimensions = research_request.get('dimensions', [])
    
    # Ensure we have at least 3 dimensions
    while len(dimensions) < 3:
        dimensions.append(f"Dimension {len(dimensions) + 1} of {research_topic}")
    
    # Replace topic and dimensions in all steps
    for i, step in enumerate(custom_workflow):
        if "content" in step:
            content = step["content"]
            content = content.replace("Agentic AI future vision", research_topic)
            content = content.replace("'Agentic AI future vision'", f"'{research_topic}'")
            
            # Replace dimension placeholders
            for j, dimension in enumerate(dimensions[:3]):
                placeholder = f"{{dimension{j+1}}}"
                if placeholder in content:
                    content = content.replace(placeholder, dimension)
            
            custom_workflow[i]["content"] = content
    
    # Handle dynamic agent steps
    for i, step in enumerate(custom_workflow):
        if step.get("type") == "dynamic" and "actions" in step:
            actions = step["actions"]
            new_actions = {}
            placeholders = list(actions.keys())
            
            for j, dimension in enumerate(dimensions[:3]):
                if j < len(placeholders):
                    new_actions[dimension] = actions[placeholders[j]]
                    if "content" in new_actions[dimension]:
                        content = new_actions[dimension]["content"]
                        for k, dim in enumerate(dimensions[:3]):
                            content = content.replace(f"{{dimension{k+1}}}", dim)
                        new_actions[dimension]["content"] = content
            
            if new_actions:
                custom_workflow[i]["actions"] = new_actions
                dimensions_str = ", ".join([f"'{d}'" for d in dimensions[:3]])
                custom_workflow[i]["initial_prompt"] = f"We will research each dimension for {research_topic}. Choose first: {dimensions_str}."
    
    # Ensure report generation is comprehensive
    for i, step in enumerate(custom_workflow):
        if step.get("agent") == "report_generator" or step.get("agent") == "comprehensive_report_generator":
            custom_workflow[i]["agent"] = "comprehensive_report_generator"
            custom_workflow[i]["content"] = f"Generate a comprehensive research report on {research_topic}."
            custom_workflow[i]["output_format"] = {
                "type": "markdown",
                "sections": [
                    "Executive Summary",
                    f"Introduction to {research_topic}",
                    "Research Methodology",
                    f"{dimensions[0]} Analysis",
                    f"{dimensions[1]} Analysis",
                    f"{dimensions[2]} Analysis", 
                    "Cross-Dimensional Insights",
                    "Conclusions and Recommendations"
                ]
            }
    
    return custom_workflow

def generate_workflow_file(research_request: Dict[str, Any], output_dir: str = None) -> str:
    """Generate the customized workflow file and return its path."""
    workflow_template = load_workflow_template()
    custom_workflow = customize_workflow(workflow_template, research_request)
    
    topic_filename = clean_topic_for_filename(research_request['topic'])
    if len(topic_filename) > 40:
        topic_filename = topic_filename[:40]
    
    timestamp = int(time.time())
    filename = f"research_workflow_{topic_filename}_{timestamp}.json"
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")
    
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(custom_workflow, f, indent=2)
    
    print(f"Generated workflow file: {output_path}")
    return output_path

def run_workflow(workflow_path: str) -> int:
    """Run the workflow using new_univ_main.py."""
    command = [sys.executable, "new_univ_main.py", "--workflow", workflow_path]
    print(f"Running command: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing workflow: {e}")
        return e.returncode

def generate_input_template(output_path: str = "research_input_template.csv"):
    """Generate a simple template input CSV file."""
    template_content = """topic,dimension1,dimension2,dimension3
Research Topic,First dimension,Second dimension,Third dimension

# Example:
# Quantum Computing in Finance,Quantum algorithms,Hardware implementation,Financial applications
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print(f"Generated input template file: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate and run a research workflow.")
    parser.add_argument('input_file', nargs='?', help='Path to the input file (CSV, JSON, or text)')
    parser.add_argument('--output-dir', help='Directory to store the generated workflow file')
    parser.add_argument('--run', action='store_true', help='Run the workflow after generating it')
    parser.add_argument('--create-template', action='store_true', help='Create a template input CSV file')
    parser.add_argument('--auto-dimensions', action='store_true', help='Automatically generate dimensions if not provided')
    args = parser.parse_args()
    
    if args.create_template:
        generate_input_template()
        if not args.input_file:
            return
    
    if not args.input_file:
        print("Error: Input file is required. Use --create-template to generate a template.")
        parser.print_help()
        return
    
    # Parse and validate input
    research_request = parse_user_input(args.input_file)
    is_valid, message = validate_input(research_request)
    
    print(f"\nResearch Topic: {research_request['topic']}")
    if research_request.get('dimensions'):
        print(f"Research Dimensions: {', '.join(research_request['dimensions'])}")
    else:
        print("Research Dimensions: None provided")
    
    # Auto-generate dimensions if needed
    if not research_request.get('dimensions') and args.auto_dimensions:
        topic = research_request['topic']
        research_request['dimensions'] = [
            f"Technical aspects of {topic}",
            f"Applications of {topic}",
            f"Future implications of {topic}"
        ]
        print(f"Auto-generated dimensions: {', '.join(research_request['dimensions'])}")
    
    if not is_valid:
        print(f"\n{message}")
        print("\nUse --auto-dimensions to use suggested dimensions.")
        return
    
    # Generate and optionally run the workflow
    workflow_path = generate_workflow_file(research_request, args.output_dir)
    if args.run:
        run_workflow(workflow_path)
    else:
        print(f"\nTo run the workflow, use: python3 new_univ_main.py --workflow {workflow_path}")

# Embedded template (minimal version of comprehensive workflow)
COMPREHENSIVE_WORKFLOW_TEMPLATE = [
    {
        "agent": "system_architect",
        "content": "Design a comprehensive research system architecture for investigating 'Agentic AI future vision'.",
        "tools": ["cognitive:list_patterns", "optimization:list_patterns", "memory:list_patterns", "planning:create_plan"],
        "output_format": {
            "type": "json",
            "schema": {
                "architecture_design": {
                    "cognitive_approach": {"selected_pattern": "string", "justification": "string"},
                    "optimization_approach": {"selected_pattern": "string", "justification": "string"},
                    "memory_system": {"selected_pattern": "string", "justification": "string"},
                    "integration_strategy": "string"
                }
            }
        }
    },
    {
        "agent": "memory_system_initializer",
        "content": "Initialize the memory system for our research on 'Agentic AI future vision'.",
        "readFrom": ["system_architect"],
        "tools": ["memory:create_system", "memory:submit_result"],
        "output_format": {
            "type": "json",
            "schema": {
                "memory_system_id": "string",
                "system_description": "string"
            }
        }
    },
    {
        "agent": "preliminary_researcher",
        "content": "Conduct comprehensive research on 'Agentic AI future vision'.",
        "readFrom": ["memory_system_initializer"],
        "tools": ["research:combined_search", "memory:store_operation"],
        "output_format": {
            "type": "json",
            "schema": {
                "initial_findings": "string",
                "key_concepts": ["string"]
            }
        }
    },
    {
        "agent": "research_focus_selector",
        "content": "Select three key dimensions of 'Agentic AI future vision' for in-depth investigation.",
        "readFrom": ["preliminary_researcher"],
        "tools": ["planning:chain_of_thought"],
        "output_format": {
            "type": "json",
            "schema": {
                "selected_dimensions": ["string"],
                "selection_rationale": "string"
            }
        }
    },
    {
        "agent": "dynamic_agent",
        "type": "dynamic",
        "initial_prompt": "We will research each of the three key dimensions. Choose the first dimension to investigate: {dimension1}, {dimension2}, or {dimension3}.",
        "readFrom": ["research_focus_selector"],
        "tools": ["planning:chain_of_thought"],
        "output_format": {
            "type": "json",
            "schema": {
                "action": "string",
                "reasoning": "string"
            }
        },
        "actions": {
            "{dimension1}": {
                "agent": "dimension1_researcher",
                "content": "Research the {dimension1} aspect of 'Agentic AI future vision'.",
                "tools": ["research:search", "memory:store_operation"]
            },
            "{dimension2}": {
                "agent": "dimension2_researcher",
                "content": "Research the {dimension2} aspect of 'Agentic AI future vision'.",
                "tools": ["research:search", "memory:store_operation"]
            },
            "{dimension3}": {
                "agent": "dimension3_researcher",
                "content": "Research the {dimension3} aspect of 'Agentic AI future vision'.",
                "tools": ["research:search", "memory:store_operation"]
            }
        }
    },
    {
        "agent": "comprehensive_report_generator",
        "content": "Generate a comprehensive research report on 'Agentic AI future vision'.",
        "readFrom": ["*"],
        "output_format": {
            "type": "markdown",
            "sections": [
                "Executive Summary",
                "Introduction",
                "Research Methodology",
                "Dimension 1 Analysis",
                "Dimension 2 Analysis",
                "Dimension 3 Analysis",
                "Conclusions and Recommendations"
            ]
        }
    }
]

if __name__ == "__main__":
    main()
