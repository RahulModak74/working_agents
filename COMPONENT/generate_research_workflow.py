#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
import subprocess
import re
from typing import Dict, Any, List, Optional

def clean_topic_for_filename(topic: str) -> str:
    """Convert a topic string into a clean filename string."""
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r'[^\w\s-]', '', topic).strip().lower()
    return re.sub(r'[-\s]+', '_', cleaned)

def create_dimension_placeholder(dimension: str) -> Dict[str, Any]:
    """Create placeholder value for dynamic dimensions in workflow."""
    # Create a placeholder that will be replaced with the actual dimension later
    safe_dimension = dimension.replace(' ', '_').lower()
    return f"{{dimension_{safe_dimension}}}"

def parse_user_input(input_file: str) -> Dict[str, Any]:
    """Parse the user input file and extract research parameters."""
    research_request = {}
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        sys.exit(1)
    
    # Determine file type and parse accordingly
    if input_file.endswith('.csv'):
        # Simple CSV parsing
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # If only one line, assume it's just the research topic
            if len(lines) == 1:
                research_request['topic'] = lines[0].strip()
                research_request['description'] = lines[0].strip()
                research_request['dimensions'] = []
            
            # If multiple lines, assume header + values format
            elif len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(',')]
                values = [v.strip() for v in lines[1].split(',')]
                
                # Create a dictionary from headers and values
                data = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
                
                # Extract key fields
                research_request['topic'] = data.get('topic', data.get('research_topic', data.get('query', '')))
                research_request['description'] = data.get('description', data.get('details', research_request['topic']))
                
                # Extract dimensions if present
                dimensions = []
                for i in range(1, 10):  # Check for dimension1, dimension2, etc.
                    dim_key = f'dimension{i}'
                    if dim_key in data and data[dim_key]:
                        dimensions.append(data[dim_key])
                
                # Alternative dimension formats
                if not dimensions and 'dimensions' in data:
                    dimensions = [d.strip() for d in data['dimensions'].split(';')]
                
                research_request['dimensions'] = dimensions
    
    elif input_file.endswith('.json'):
        # JSON parsing
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            research_request['topic'] = data.get('topic', data.get('research_topic', data.get('query', '')))
            research_request['description'] = data.get('description', data.get('details', research_request['topic']))
            research_request['dimensions'] = data.get('dimensions', [])
    
    else:
        # Treat as plain text
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            research_request['topic'] = content.split('\n')[0] if '\n' in content else content
            research_request['description'] = content
            research_request['dimensions'] = []
    
    # Ensure we have at least a topic
    if not research_request.get('topic'):
        print("Error: Could not extract a research topic from the input file.")
        sys.exit(1)
    
    return research_request

def load_workflow_template() -> List[Dict[str, Any]]:
    """Load the base workflow template."""
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "cognitive_optimized_memory_research_workflow_template.json")
    
    # If template doesn't exist, use a minimal template
    if not os.path.exists(template_path):
        print(f"Warning: Template file {template_path} not found. Using embedded template.")
        
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Check if we have the template in different locations
        alternate_paths = [
            os.path.join(script_dir, "templates", "cognitive_optimized_memory_research_workflow.json"),
            os.path.join(script_dir, "..", "templates", "cognitive_optimized_memory_research_workflow.json"),
        ]
        
        for alt_path in alternate_paths:
            if os.path.exists(alt_path):
                template_path = alt_path
                break
        
        # If still not found, use embedded minimal template
        if not os.path.exists(template_path):
            return DEFAULT_WORKFLOW_TEMPLATE
    
    # Load the template
    with open(template_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def customize_workflow(workflow: List[Dict[str, Any]], research_request: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Customize the workflow with the research parameters."""
    # Make a deep copy to avoid modifying the original
    custom_workflow = json.loads(json.dumps(workflow))
    
    research_topic = research_request['topic']
    description = research_request['description']
    dimensions = research_request['dimensions']
    
    # Fill in dimensions if provided, otherwise leave for the workflow to determine
    dimension_placeholders = {
        "{dimension1}": dimensions[0] if len(dimensions) > 0 else None,
        "{dimension2}": dimensions[1] if len(dimensions) > 1 else None,
        "{dimension3}": dimensions[2] if len(dimensions) > 2 else None
    }
    
    # Replace topic and description in all steps
    for i, step in enumerate(custom_workflow):
        if "content" in step:
            # Replace the 'Agentic AI future vision' in the template with the actual topic
            content = step["content"]
            content = content.replace("Agentic AI future vision", research_topic)
            content = content.replace("'Agentic AI future vision'", f"'{research_topic}'")
            
            # Replace dimension placeholders if available
            for placeholder, value in dimension_placeholders.items():
                if value and placeholder in content:
                    content = content.replace(placeholder, value)
            
            custom_workflow[i]["content"] = content
    
    # For dynamic agent step, handle dimension actions
    for i, step in enumerate(custom_workflow):
        if step.get("type") == "dynamic" and "actions" in step:
            actions = step["actions"]
            
            # If dimensions were provided, use them for the actions
            new_actions = {}
            
            if len(dimensions) >= 3:
                # Replace placeholder keys with actual dimensions
                for j, (placeholder, dimension) in enumerate(zip(actions.keys(), dimensions[:3])):
                    new_actions[dimension] = actions[placeholder]
                    
                    # Update the content to mention this specific dimension
                    if "content" in new_actions[dimension]:
                        content = new_actions[dimension]["content"]
                        content = content.replace("{dimension1}", dimension)
                        content = content.replace("{dimension2}", dimension)
                        content = content.replace("{dimension3}", dimension)
                        new_actions[dimension]["content"] = content
                
                custom_workflow[i]["actions"] = new_actions
                
                # Update the initial prompt to list the specific dimensions
                dimensions_str = ", ".join([f"'{d}'" for d in dimensions[:3]])
                initial_prompt = f"We will conduct in-depth research on each of the three key dimensions selected for {research_topic}. Choose the first dimension to investigate: {dimensions_str}."
                custom_workflow[i]["initial_prompt"] = initial_prompt
    
    # Add reflection steps after each major research step
    enhanced_workflow = []
    for step in custom_workflow:
        enhanced_workflow.append(step)
        
        # Add reflection steps after dimension researchers
        if step.get("agent", "").endswith("_researcher") and "dimension" in step.get("agent", ""):
            researcher_name = step.get("agent", "")
            
            # Add metacognitive analyzer
            metacog_analyzer = {
                "agent": f"{researcher_name}_metacognitive_analyzer",
                "content": f"Analyze the research on {research_topic} conducted by {researcher_name} to identify reasoning flaws, biases, and improvement opportunities. Apply rigorous metacognitive reflection to enhance research quality.",
                "readFrom": [researcher_name],
                "tools": ["cognitive:create_session", "planning:chain_of_thought"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "identified_biases": ["string"],
                        "logical_flaws": ["string"],
                        "improvement_directives": ["string"],
                        "enhanced_prompting_strategy": "string"
                    }
                }
            }
            enhanced_workflow.append(metacog_analyzer)
            
            # Add refined researcher
            refined_researcher = {
                "agent": f"{researcher_name}_refined",
                "content": f"Refine and improve the research on {research_topic} using the metacognitive analysis. Address the identified issues, apply the enhanced prompting strategy, and produce a significantly improved version of the research.",
                "readFrom": [researcher_name, f"{researcher_name}_metacognitive_analyzer"],
                "tools": ["research:search", "research:fetch_content", "research:analyze_content", "memory:store_operation"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "refined_findings": "string",
                        "improvement_summary": "string",
                        "confidence_assessment": "number"
                    }
                }
            }
            enhanced_workflow.append(refined_researcher)
    
    # Add a final quality assessment step before the report generation
    quality_assessment = {
        "agent": "quality_assessment_agent",
        "content": f"Conduct a comprehensive assessment of the research quality on {research_topic}. Evaluate the effectiveness of the metacognitive reflection cycles, the depth and breadth of the research, and the coherence of the integrated findings.",
        "readFrom": ["*"],
        "tools": ["planning:chain_of_thought"],
        "output_format": {
            "type": "json",
            "schema": {
                "quality_metrics": {
                    "depth_score": "number",
                    "breadth_score": "number",
                    "coherence_score": "number",
                    "insight_novelty_score": "number",
                    "overall_quality_score": "number"
                },
                "reflection_effectiveness": "string",
                "most_valuable_insights": ["string"],
                "final_improvement_recommendations": ["string"]
            }
        }
    }
    
    # Find the right position for this (before the comprehensive report generator)
    for i, step in enumerate(enhanced_workflow):
        if step.get("agent") == "comprehensive_report_generator":
            enhanced_workflow.insert(i, quality_assessment)
            break
    else:
        # If not found, just add it at the end
        enhanced_workflow.append(quality_assessment)
    
    return enhanced_workflow

def generate_workflow_file(research_request: Dict[str, Any], output_dir: str = None) -> str:
    """Generate the customized workflow file and return its path."""
    # Load the template
    workflow_template = load_workflow_template()
    
    # Customize the workflow
    custom_workflow = customize_workflow(workflow_template, research_request)
    
    # Create a filename based on the topic
    topic_filename = clean_topic_for_filename(research_request['topic'])
    if len(topic_filename) > 40:
        topic_filename = topic_filename[:40]
    
    timestamp = int(time.time())
    filename = f"research_workflow_{topic_filename}_{timestamp}.json"
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Write the workflow to a file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(custom_workflow, f, indent=2)
    
    print(f"Generated workflow file: {output_path}")
    return output_path

def run_workflow(workflow_path: str) -> int:
    """Run the workflow using new_univ_main.py."""
    # Build the command
    command = [sys.executable, "new_univ_main.py", "--workflow", workflow_path]
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        # Run the command
        result = subprocess.run(command, check=True)
        print(f"Workflow execution completed with return code: {result.returncode}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error executing workflow: {e}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="Generate and run a research workflow based on user input.")
    parser.add_argument('input_file', help='Path to the input file (CSV, JSON, or text)')
    parser.add_argument('--output-dir', help='Directory to store the generated workflow file')
    parser.add_argument('--run', action='store_true', help='Run the workflow after generating it')
    args = parser.parse_args()
    
    # Parse the user input
    research_request = parse_user_input(args.input_file)
    
    # Generate the workflow file
    workflow_path = generate_workflow_file(research_request, args.output_dir)
    
    # Run the workflow if requested
    if args.run:
        run_workflow(workflow_path)
    else:
        print(f"To run the workflow, use: python3 new_univ_main.py --workflow {workflow_path}")

# Default minimal workflow template
DEFAULT_WORKFLOW_TEMPLATE = [
    {
        "agent": "system_architect",
        "content": "Design a comprehensive research system architecture for investigating 'Agentic AI future vision'. Determine the optimal combination of cognitive patterns, optimization approaches, and memory systems to create a sophisticated, self-improving research framework.",
        "tools": ["cognitive:list_patterns", "optimization:list_patterns", "memory:list_patterns", "planning:create_plan"],
        "output_format": {
            "type": "json",
            "schema": {
                "architecture_design": {
                    "cognitive_approach": {
                        "selected_pattern": "string",
                        "justification": "string"
                    },
                    "optimization_approach": {
                        "selected_pattern": "string",
                        "justification": "string"
                    },
                    "memory_system": {
                        "selected_pattern": "string",
                        "justification": "string"
                    },
                    "integration_strategy": "string",
                    "expected_synergies": ["string"]
                }
            }
        }
    },
    {
        "agent": "preliminary_researcher",
        "content": "Conduct preliminary research on 'Agentic AI future vision'.",
        "tools": ["research:combined_search"],
        "output_format": {
            "type": "json",
            "schema": {
                "initial_findings": "string",
                "key_concepts": ["string"]
            }
        }
    },
    {
        "agent": "report_generator",
        "content": "Generate a comprehensive report based on the research findings.",
        "readFrom": ["preliminary_researcher"],
        "output_format": {
            "type": "markdown",
            "sections": [
                "Executive Summary",
                "Key Findings",
                "Recommendations"
            ]
        }
    }
]

if __name__ == "__main__":
    main()
