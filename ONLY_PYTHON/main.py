# main.py
# Example usage of the agent systems

import logging
import json
from typing import Dict, Any

# Import our agent system modules
from agent_framework import AgentSystem, Agent, DynamicAgent
from agent_tools import ToolRegistry
from agent_system_types import (
    AdvancedCognitionSystem,
    DebateSystem,
    MetacognitiveSystem,
    TreeOfThoughtsSystem,
    DistributedMemorySystem
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_results(title: str, results: Dict[str, Any]) -> None:
    """
    Pretty print the results of an agent system execution.
    
    Args:
        title: Title for the results
        results: The results to print
    """
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)
    
    # Don't print the raw_results to keep output cleaner
    if "raw_results" in results:
        result_copy = results.copy()
        del result_copy["raw_results"]
    else:
        result_copy = results
    
    # Convert to JSON and pretty print
    print(json.dumps(result_copy, indent=2))
    print("=" * 80 + "\n")

def advanced_cognition_example() -> None:
    """
    Example using the Advanced Cognition System.
    """
    system = AdvancedCognitionSystem()
    
    problem_description = """
    Design a sustainable urban transportation system for a mid-sized city (population 500,000) 
    that reduces carbon emissions by 50% within 10 years while improving accessibility and 
    affordability for all residents. The solution should consider infrastructure changes, 
    technology adoption, policy modifications, and behavior change strategies.
    """
    
    results = system.solve_problem(problem_description)
    print_results("Advanced Cognition System Results", results)

def debate_example() -> None:
    """
    Example using the Debate System.
    """
    system = DebateSystem()
    
    topic_description = """
    Should artificial general intelligence (AGI) development be regulated by international 
    governing bodies? Consider aspects of safety, innovation, economic impacts, and global 
    power dynamics.
    """
    
    results = system.debate_topic(topic_description)
    print_results("Debate System Results", results)

def metacognitive_example() -> None:
    """
    Example using the Metacognitive System.
    """
    system = MetacognitiveSystem()
    
    problem_description = """
    A city is experiencing frequent flooding during heavy rainfall. The current stormwater 
    system cannot handle the increased precipitation caused by climate change. Design a 
    solution that will prevent flooding, improve water management, and be cost-effective 
    for the city with a limited budget of $50 million.
    """
    
    results = system.solve_with_metacognition(problem_description)
    print_results("Metacognitive System Results", results)

def tree_of_thoughts_example() -> None:
    """
    Example using the Tree of Thoughts System.
    """
    system = TreeOfThoughtsSystem()
    
    problem_description = """
    Design an educational intervention to improve science literacy and critical thinking 
    skills among middle school students, particularly in underserved communities. The 
    solution should be scalable, engage students effectively, and improve measurable 
    outcomes in both science understanding and general critical reasoning.
    """
    
    results = system.solve_with_tot(problem_description)
    print_results("Tree of Thoughts System Results", results)

def distributed_memory_example() -> None:
    """
    Example using the Distributed Memory System.
    """
    system = DistributedMemorySystem()
    
    application_domain = "healthcare data management system for a network of hospitals"
    
    results = system.design_memory_system(application_domain)
    print_results("Distributed Memory System Results", results)

if __name__ == "__main__":
    # Run the examples
    print("Running agent system examples...")
    
    # Choose which examples to run
    examples = {
        "advanced_cognition": True,
        "debate": True,
        "metacognitive": True,
        "tree_of_thoughts": True,
        "distributed_memory": True
    }
    
    if examples["advanced_cognition"]:
        advanced_cognition_example()
    
    if examples["debate"]:
        debate_example()
    
    if examples["metacognitive"]:
        metacognitive_example()
    
    if examples["tree_of_thoughts"]:
        tree_of_thoughts_example()
    
    if examples["distributed_memory"]:
        distributed_memory_example()
    
    print("All examples completed!")
