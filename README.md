# Enhanced Multi-Agent System

A modular system for orchestrating AI agents with persistent memory, dynamic behavior, and structured output parsing. Extremely simple and inbuilt support for CyberSecurity.

Features: Memory, Dynamic Agents, Strcutured Output parsing

To Add:Tool integration, streaming, chain types & reasoning patterns, streaming, vectorization and embeddings.

## Features

1. **Memory Management**
   - Persistent conversation context between agents and across sessions
   - SQLite-based memory storage with flexible querying
   - Memory contexts automatically added to prompts

2. **Dynamic Agent Behavior**
   - Agents can choose their next actions based on task requirements
   - Branching workflows that adapt based on agent decisions
   - Decision-based delegation to specialized agents

3. **Structured Output Parsing**
   - JSON schema validation for structured agent outputs
   - Markdown section validation for document outputs
   - Automatic extraction and structuring of agent responses

## Project Structure

```
multi-agent-system/
├── __init__.py             # Package initialization
├── config.py               # Configuration settings
├── memory_manager.py       # Memory persistence system
├── output_parser.py        # Structured output parsing
├── agent.py                # Base agent implementation
├── dynamic_agent.py        # Dynamic decision-making agent
├── agent_system.py         # Agent orchestration system
├── cli.py                  # Command-line interface
├── main.py                 # Main application entry point
└── example_workflow.json   # Example workflow configuration
```

## Installation

```bash
# Clone the repository
git clone https://github.com/RahulModak74/enhanced_agents.git
cd multi-agent-system

# Install dependencies
pip install jsonschema

# Make the main script executable
chmod +x main.py
```

## Usage

### Interactive Mode

```bash
./main.py
```

This launches the interactive command-line interface where you can:

- Create agents: `create agent1 [model] [type]`
- Run agents: `run agent1 What is the risk score? [file] data.csv [memory:risk] [format:json]`
- Execute workflows: `workflow example_workflow.json`
- Access memory: `memory get risk_assessment` or `memory list`
- List agents: `list`
- Get agent outputs: `get agent1`

### Workflow Mode

```bash
./main.py --workflow example_workflow.json
```

This executes a pre-defined workflow directly without entering the interactive shell.

## Workflow Configuration

Workflows are defined as JSON arrays of agent execution steps:

```json
[
  {
    "agent": "agent1",
    "content": "Analyze this data...",
    "file": "data.csv",
    "output_format": {
      "type": "json",
      "schema": {
        "score": "number",
        "reasoning": "string"
      }
    },
    "memory_id": "analysis"
  },
  {
    "agent": "dynamic_agent",
    "type": "dynamic",
    "initial_prompt": "Decide what to do next...",
    "readFrom": ["agent1"],
    "memory_id": "analysis",
    "actions": {
      "action1": {
        "agent": "agent2",
        "content": "Follow-up task..."
      },
      "action2": {
        "agent": "agent3",
        "content": "Alternative task..."
      }
    }
  }
]
```

## Memory System

The memory system uses SQLite to store:

- Agent outputs with metadata and timestamps
- Conversation contexts for multi-turn interactions
- Memory contexts that can be shared across agents

## Extending the System

### Adding New Agent Types

Create a new class that inherits from `Agent` or `DynamicAgent`:

```python
class SpecializedAgent(Agent):
    def __init__(self, name, model=CONFIG["default_model"], memory_manager=None):
        super().__init__(name, model, memory_manager)
        # Custom initialization
    
    def special_execution(self, content, **kwargs):
        # Custom execution logic
        return result
```

### Adding New Output Formats

Extend the `OutputParser` class with new format handlers:

```python
@staticmethod
def extract_custom_format(text, output_format):
    # Custom format extraction logic
    return structured_data
```

## License

SSPL
