# Urban Planning Multi-Agent System Example

This example demonstrates a complex use case of multiple agent systems working together to solve an urban planning challenge. It uses the configuration settings from `config.py` and integrates with external LLM APIs.

## Project Structure

The project consists of several modules:

1. **agent_framework.py**: Core framework for agents and agent systems
2. **agent_tools.py**: Implementation of tools that agents can use
3. **agent_system_types.py**: Specialized agent system implementations
4. **complex_urban_planning.py**: Example integration of multiple agent systems for urban planning

## Prerequisites

- Python 3.7+
- SQLite3
- OpenRouter API key (or modify to use a different LLM provider)
- Required Python packages:
  - requests

## Setup

1. Ensure your `config.py` file is correctly set up with API keys and paths
2. Create the necessary directories:
   ```bash
   mkdir -p ./agent_outputs
   mkdir -p ./agent_memory
   ```
3. Install required packages:
   ```bash
   pip install requests
   ```

## Running the Example

To run the urban planning example:

```bash
# Navigate to your project directory
cd your_project_directory

# Run the example script
python complex_urban_planning.py
```

The script will:
1. Set up a SQLite database with urban planning data
2. Coordinate multiple agent systems to solve the challenge
3. Save results to the configured output directory
4. Display a summary of the results

## Expected Output

The script will create:
1. A log file (`urban_planning.log`) in the output directory
2. A solution file (`urban_planning_solution.json`) with the comprehensive solution
3. A SQLite database with urban planning data

The terminal output will show:
- A summary of the challenge being solved
- Progress updates from each agent system
- A synthesis of key perspectives
- Execution times for each system component
- Paths to the output files

## Customizing the Example

To modify the urban planning challenge:
1. Edit the `challenge` variable in the `main()` function
2. Adjust the database schema or sample data in the `setup_database()` method
3. Configure different LLM parameters in the `llm_query()` function

## Integration with OpenRouter API

The example uses OpenRouter to access various language models. To use a different model:
1. Update the `default_model` in your `config.py` file
2. Make sure your API key has access to the specified model

## Extending the Example

You can extend this example by:
1. Adding more agent systems to the coordination process
2. Enhancing the database schema with more urban planning data
3. Creating additional specialized tools for urban planning tasks
4. Implementing visualization of the solution components
5. Adding a web interface to interact with the urban planning system

## Troubleshooting

If you encounter issues:
1. Check the log file for detailed error messages
2. Verify your API key is valid
3. Ensure the database file permissions are correct
4. Check that all directories specified in the config exist
