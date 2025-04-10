# Multi-Agent Cognitive System Framework

This framework provides a Python implementation of various advanced cognitive agent systems based on JSON configurations. It allows for the creation and execution of sophisticated multi-agent systems for problem-solving, debate, metacognitive reflection, and more.

## Project Structure

The project consists of several modules:

1. **agent_framework.py**: Core framework for agents and agent systems
2. **agent_tools.py**: Implementation of tools that agents can use
3. **agent_system_types.py**: Specialized agent system implementations
4. **main.py**: Example usage of the agent systems

## Agent Types

- **Agent**: Base agent class that follows a defined process and produces outputs
- **DynamicAgent**: Advanced agent that can select from multiple actions

## Agent Systems

The framework includes several specialized agent systems:

1. **AdvancedCognitionSystem**: Determines the best cognitive approach for a problem and applies it
2. **DebateSystem**: Conducts a structured debate on a topic with agents representing different positions
3. **MetacognitiveSystem**: Solves a problem, then analyzes and improves the solution process
4. **TreeOfThoughtsSystem**: Generates multiple solution paths and explores the most promising ones
5. **DistributedMemorySystem**: Creates and manages a sophisticated memory architecture

## Tools

Agents can use various tools to accomplish their tasks:

- **chain_of_thought**: Implements a chain-of-thought reasoning process
- **create_plan**: Creates a structured plan to achieve an objective
- **get_summary**: Generates a concise summary of content
- **sql_query**: Simulates executing SQL queries
- **vector_db_add**, **vector_db_batch_add**, **vector_db_search**: Simulates vector database operations

## Usage Example

```python
from agent_system_types import AdvancedCognitionSystem

# Create an Advanced Cognition System
system = AdvancedCognitionSystem()

# Define a problem to solve
problem_description = """
Design a sustainable urban transportation system for a mid-sized city (population 500,000) 
that reduces carbon emissions by 50% within 10 years while improving accessibility and 
affordability for all residents.
"""

# Solve the problem
results = system.solve_problem(problem_description)

# Use the results
print(f"Selected approach: {results['approach_used']}")
print(f"Solution: {results['solution']}")
```

## Getting Started

1. Clone this repository
2. Make sure you have Python 3.7+ installed
3. Run `python main.py` to see the examples in action

## Extending the Framework

### Adding New Tools

Add new tool functions to `agent_tools.py` and register them in the `ToolRegistry`:

```python
def my_new_tool(input_data, context=None):
    # Tool implementation
    return {"result": "processed data"}

# Register the tool
ToolRegistry.register_tool("category:my_new_tool", my_new_tool)
```

### Creating Custom Agent Systems

Extend the `AgentSystem` class to create your own agent system:

```python
class MyCustomSystem(AgentSystem):
    def __init__(self):
        super().__init__("My Custom System")
        
        # Define your agent configurations
        configs = [
            # Agent configs here
        ]
        
        # Add the agents to the system
        self.add_agents_from_config(configs)
    
    def my_custom_method(self, input_data):
        # Custom system functionality
        results = self.execute_workflow("starting_agent", {"input": input_data})
        return results
```

## Notes

- This is a simulated implementation that provides the structure and flow of a multi-agent system
- In a real-world implementation, the tools would connect to actual services, databases, and AI models
- The agents would use sophisticated reasoning, planning, and decision-making capabilities

## License

This project is available under the MIT License.
