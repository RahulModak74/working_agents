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
            lines = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
            
            # If only one line, assume it's just the research topic
            if len(lines) == 1:
                research_request['topic'] = lines[0].strip()
                research_request['dimensions'] = []
            
            # Multiple lines - header + values format
            elif len(lines) > 1:
                headers = [h.strip() for h in lines[0].split(',')]
                
                # Skip the example line if it contains placeholder text as the topic value
                data_line_index = 1
                values = [v.strip() for v in lines[data_line_index].split(',')]
                if values[0] in ["Research Topic", "YOUR RESEARCH TOPIC HERE"] and len(lines) > 2:
                    data_line_index = 2
                    values = [v.strip() for v in lines[data_line_index].split(',')]
                
                data = {headers[i]: values[i] for i in range(min(len(headers), len(values)))}
                research_request['topic'] = data.get('topic', '')
                
                # Extract dimensions - support up to 10
                dimensions = []
                for i in range(1, 11):
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
    
    # Ensure we have exactly 10 dimensions - no more, no less
    while len(dimensions) < 10:
        dimensions.append(f"Dimension {len(dimensions) + 1} of {research_topic}")
    
    # Limit to exactly 10 dimensions
    dimensions = dimensions[:10]
    
    # Replace topic and dimensions in all steps
    for i, step in enumerate(custom_workflow):
        if "content" in step:
            content = step["content"]
            content = content.replace("Agentic AI future vision", research_topic)
            content = content.replace("'Agentic AI future vision'", f"'{research_topic}'")
            content = content.replace("Research Topic", research_topic)
            content = content.replace("'Research Topic'", f"'{research_topic}'")
            
            # Replace dimension placeholders
            for j, dimension in enumerate(dimensions):
                placeholder = f"{{dimension{j+1}}}"
                if placeholder in content:
                    content = content.replace(placeholder, dimension)
            
            custom_workflow[i]["content"] = content
    
    # Remove any dimension selection steps that would limit to just 3 dimensions
    custom_workflow = [step for step in custom_workflow 
                     if not any(x in step.get("agent", "") for x in 
                              ["second_dimension_selector", "third_dimension_selector"])]
    
    # Create 10 dedicated dimension researcher agents
    dimension_researchers = []
    for i, dimension in enumerate(dimensions):
        researcher = {
            "agent": f"dimension{i+1}_researcher",
            "content": f"Conduct comprehensive research on the '{dimension}' aspect of '{research_topic}'. Focus on collecting authoritative sources, key insights, and relevant data.",
            "readFrom": ["preliminary_researcher", "research_focus_selector"],
            "tools": ["research:search", "research:cite_sources", "research:fetch_content", "memory:store_operation"],
            "output_format": {
                "type": "json",
                "schema": {
                    "dimension_findings": "string",
                    "key_insights": ["string"],
                    "sources": [{
                        "title": "string", 
                        "url": "string", 
                        "authors": ["string"],
                        "relevance": "string"
                    }]
                }
            }
        }
        dimension_researchers.append(researcher)
    
    # Find position to insert dimension researchers
    dynamic_agent_index = next((i for i, step in enumerate(custom_workflow) 
                           if step.get("type") == "dynamic"), -1)
    
    if dynamic_agent_index != -1:
        # Insert all dimension researchers after the dynamic agent
        for i, researcher in enumerate(dimension_researchers):
            custom_workflow.insert(dynamic_agent_index + 1 + i, researcher)
    
    # Modify the dynamic agent to handle all 10 dimensions
    for i, step in enumerate(custom_workflow):
        if step.get("type") == "dynamic" and "actions" in step:
            # Create actions for all 10 dimensions
            new_actions = {}
            for j, dimension in enumerate(dimensions):
                new_actions[dimension] = {
                    "agent": f"dimension{j+1}_action",
                    "content": f"Research the '{dimension}' aspect of '{research_topic}' with proper citations and link tracking.",
                    "tools": ["research:search", "research:cite_sources", "memory:store_operation"],
                    "output_format": {
                        "type": "json",
                        "schema": {
                            "findings": "string",
                            "sources": [{
                                "title": "string", 
                                "url": "string", 
                                "relevance": "string"
                            }]
                        }
                    }
                }
            
            # Replace existing actions with our new ones
            custom_workflow[i]["actions"] = new_actions
            dimensions_str = ", ".join([f"'{d}'" for d in dimensions])
            custom_workflow[i]["initial_prompt"] = f"We will research each dimension for {research_topic}. Choose first from: {dimensions_str}."
    
    # Add a source collector agent
    source_collector = {
        "agent": "source_collector",
        "content": f"Collect and validate all sources used in researching each dimension of '{research_topic}'. Ensure proper citation format and link tracking.",
        "readFrom": ["*"],
        "tools": ["research:validate_sources", "research:format_citations", "memory:retrieve_all"],
        "output_format": {
            "type": "json",
            "schema": {
                "source_collection": [{
                    "dimension": "string",
                    "sources": [{
                        "title": "string",
                        "url": "string",
                        "authors": ["string"],
                        "publication_date": "string",
                        "citation_format": "string"
                    }]
                }]
            }
        }
    }
    
    # Add a cross-dimensional analysis agent
    cross_analysis = {
        "agent": "cross_dimensional_analyzer",
        "content": f"Analyze relationships and patterns across all ten dimensions of '{research_topic}'. Draw connections between dimensions and identify emergent insights.",
        "readFrom": dimension_researchers + ["*"],
        "tools": ["memory:retrieve_all", "planning:chain_of_thought"],
        "output_format": {
            "type": "json",
            "schema": {
                "cross_connections": [{
                    "dimensions_involved": ["string"],
                    "relationship": "string",
                    "insight": "string"
                }],
                "emergent_themes": ["string"],
                "meta_analysis": "string"
            }
        }
    }
    
    # Find position to insert the source collector and cross-dimensional analyzer
    report_gen_index = next((i for i, step in enumerate(custom_workflow) 
                         if step.get("agent") == "comprehensive_report_generator"), -1)
    
    if report_gen_index != -1:
        custom_workflow.insert(report_gen_index, source_collector)
        custom_workflow.insert(report_gen_index, cross_analysis)
    
    # Update the comprehensive report generator to include all 10 dimensions
    for i, step in enumerate(custom_workflow):
        if step.get("agent") == "comprehensive_report_generator" or step.get("agent") == "report_generator":
            step["agent"] = "comprehensive_report_generator"
            step["content"] = f"Generate a comprehensive research report on '{research_topic}' with citations to all sources used. Include analysis of all ten dimensions and cross-dimensional insights."
            
            # Create sections for all 10 dimensions
            dimension_sections = [f"{dim} Analysis" for dim in dimensions]
            
            step["output_format"] = {
                "type": "markdown",
                "sections": [
                    "Executive Summary",
                    f"Introduction to {research_topic}",
                    "Research Methodology",
                ] + dimension_sections + [
                    "Cross-Dimensional Insights",
                    "Conclusions and Recommendations",
                    "References and Sources"
                ]
            }
            
            # Ensure it uses citation tools
            step["tools"] = ["research:format_citations", "memory:retrieve_all"]
            step["readFrom"] = ["*"]
    
    # Replace any references to "First dimension", "Second dimension", etc. in all steps
    ordinal_names = ["First dimension", "Second dimension", "Third dimension", 
                    "Fourth dimension", "Fifth dimension", "Sixth dimension", 
                    "Seventh dimension", "Eighth dimension", "Ninth dimension", 
                    "Tenth dimension"]
    
    for i, step in enumerate(custom_workflow):
        if "content" in step:
            content = step["content"]
            for j, dimension in enumerate(dimensions):
                if j < len(ordinal_names):
                    content = content.replace(ordinal_names[j], dimension)
            custom_workflow[i]["content"] = content
    
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

def main():
    parser = argparse.ArgumentParser(description="Generate a 10-dimension research workflow that includes sources and links.")
    parser.add_argument('input_file', help='Path to the input file (CSV or JSON)')
    parser.add_argument('--output-dir', help='Directory to store the generated workflow file')
    parser.add_argument('--run', action='store_true', help='Run the workflow after generating it')
    args = parser.parse_args()
    
    # Parse and validate input
    research_request = parse_user_input(args.input_file)
    is_valid, message = validate_input(research_request)
    
    print(f"\nResearch Topic: {research_request['topic']}")
    if research_request.get('dimensions'):
        print(f"\nResearch Dimensions:")
        for i, dim in enumerate(research_request['dimensions']):
            print(f"{i+1}. {dim}")
    else:
        print("Warning: No dimensions provided in input file.")
    
    if not is_valid:
        print(f"\n{message}")
        return
    
    # Generate and optionally run the workflow
    workflow_path = generate_workflow_file(research_request, args.output_dir)
    if args.run:
        run_workflow(workflow_path)
    else:
        print(f"\nTo run the workflow, use: python3 new_univ_main.py --workflow {workflow_path}")

# Embedded template with support for 10 dimensions and source tracking
COMPREHENSIVE_WORKFLOW_TEMPLATE = [
    {
        "agent": "system_architect",
        "content": "Design a comprehensive research system architecture for investigating 'Research Topic' across all 10 dimensions. Ensure the system supports extensive link tracking and source citation.",
        "tools": ["cognitive:list_patterns", "optimization:list_patterns", "memory:list_patterns", "planning:create_plan"],
        "output_format": {
            "type": "json",
            "schema": {
                "architecture_design": {
                    "cognitive_approach": {"selected_pattern": "string", "justification": "string"},
                    "optimization_approach": {"selected_pattern": "string", "justification": "string"},
                    "memory_system": {"selected_pattern": "string", "justification": "string"},
                    "integration_strategy": "string",
                    "citation_tracking": {"approach": "string", "implementation": "string"}
                }
            }
        }
    },
    {
        "agent": "memory_system_initializer",
        "content": "Initialize the memory system for our research on 'Research Topic' with robust citation tracking and link storage capabilities.",
        "readFrom": ["system_architect"],
        "tools": ["memory:create_system", "memory:submit_result"],
        "output_format": {
            "type": "json",
            "schema": {
                "memory_system_id": "string",
                "system_description": "string",
                "citation_tracking": {"enabled": "boolean", "methodology": "string"}
            }
        }
    },
    {
        "agent": "preliminary_researcher",
        "content": "Conduct preliminary research on 'Research Topic' with a focus on identifying authoritative sources for all 10 dimensions. Store all sources with complete citation information.",
        "readFrom": ["memory_system_initializer"],
        "tools": ["research:combined_search", "research:cite_sources", "memory:store_operation"],
        "output_format": {
            "type": "json",
            "schema": {
                "initial_findings": "string",
                "key_concepts": ["string"],
                "primary_sources": [{"title": "string", "url": "string", "relevance": "string"}]
            }
        }
    },
    {
        "agent": "research_focus_selector",
        "content": "Review our preliminary research on 'Research Topic' and confirm the 10 dimensions for in-depth investigation. These will be studied in parallel rather than sequentially.",
        "readFrom": ["preliminary_researcher"],
        "tools": ["planning:chain_of_thought", "memory:retrieve_all"],
        "output_format": {
            "type": "json",
            "schema": {
                "confirmed_dimensions": ["string"],
                "research_approach": "string"
            }
        }
    },
    {
        "agent": "dynamic_agent",
        "type": "dynamic",
        "initial_prompt": "We will research each of the ten dimensions of 'Research Topic'. Choose one dimension to investigate first.",
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
                "content": "Research the {dimension1} aspect of 'Research Topic' with proper citations.",
                "tools": ["research:search", "research:cite_sources", "memory:store_operation"]
            },
            "{dimension2}": {
                "agent": "dimension2_researcher",
                "content": "Research the {dimension2} aspect of 'Research Topic' with proper citations.",
                "tools": ["research:search", "research:cite_sources", "memory:store_operation"]
            }
        }
    },
    {
        "agent": "comprehensive_report_generator",
        "content": "Generate a comprehensive research report on 'Research Topic' with proper citations to all sources used.",
        "readFrom": ["*"],
        "tools": ["research:format_citations", "memory:retrieve_all"],
        "output_format": {
            "type": "markdown",
            "sections": [
                "Executive Summary",
                "Introduction",
                "Research Methodology",
                "Dimension 1 Analysis",
                "Dimension 2 Analysis",
                "Dimension 3 Analysis",
                "Dimension 4 Analysis",
                "Dimension 5 Analysis",
                "Dimension 6 Analysis",
                "Dimension 7 Analysis",
                "Dimension 8 Analysis",
                "Dimension 9 Analysis",
                "Dimension 10 Analysis",
                "Cross-Dimensional Insights",
                "Conclusions and Recommendations",
                "References and Sources"
            ]
        }
    }
]

if __name__ == "__main__":
    main()
