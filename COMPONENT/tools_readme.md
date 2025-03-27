# Enhanced Multi-Agent System Tools Extension

This extension adds powerful tools capabilities to the Enhanced Multi-Agent System, including:

1. **SQL Database Tools**: Query and manipulate SQL databases
2. **HTTP/REST API Tools**: Make requests to external APIs and web services
3. **Vector Database Tools**: Create semantic search capabilities using embeddings
4. **Planning tools** - Chain of Thought, ReAct (Reasoning + Acting)

## Installation

1. Install the additional requirements:

```bash
pip install -r requirements_tools.txt
```

2. Copy the new tool files to your project directory:
   - `sql_tool.py`
   - `http_tool.py`
   - `vector_db.py`
   - `tools.py`

3. Apply the code changes from `agent_tools_integration.py` to the following files:
   - `agent.py`
   - `dynamic_agent.py`
   - `agent_system.py`
   - `cli.py`
   - `main.py`

## Using Tools in Workflows

Tools can be used in workflow JSON files by adding a `tools` array to each agent step:

```json
{
  "agent": "data_analyzer",
  "content": "Analyze the database data",
  "tools": ["sql:query", "sql:tables", "sql:schema"],
  "memory_id": "data_analysis"
}
```

## Available Tools

### SQL Tools

- `sql:query` - Execute SQL queries
- `sql:tables` - List tables in a database
- `sql:schema` - Get database schema information

### HTTP Tools

- `http:get` - Make HTTP GET requests
- `http:post` - Make HTTP POST requests with JSON data
- `http:put` - Make HTTP PUT requests
- `http:delete` - Make HTTP DELETE requests

### Vector Database Tools

- `vector_db:search` - Search for similar documents
- `vector_db:add` - Add a single document to the vector database
- `vector_db:batch_add` - Add multiple documents to the vector database

## Command Line Usage

The CLI has been extended with new commands for using tools:

```bash
# Execute a single tool with an agent
> tool agent1 sql:query query="SELECT * FROM users"

# Run an agent with access to specific tools
> tool_run agent1 "Analyze the database and search for similar patterns" [tool:sql:query,vector_db:search] [memory:analysis]
```

## Tool Workflow Example

You can run a workflow that uses tools with:

```bash
python main.py --tool-workflow tool_workflow_example.json
```

## Creating Custom Tool Configurations

### SQL Database Configuration

```python
from tools import TOOL_MANAGER

# Create a new SQL tool for a specific database
TOOL_MANAGER.create_sql_tool("users_db", "databases/users.db")

# Use the SQL tool in your code
sql_tool = TOOL_MANAGER.get_sql_tool("users_db")
success, results = sql_tool.execute_query("SELECT * FROM users")
```

### HTTP API Configuration

```python
from tools import TOOL_MANAGER

# Create an HTTP tool for a specific API
TOOL_MANAGER.create_http_tool(
    "weather_api", 
    "https://api.weather.com",
    {"Content-Type": "application/json"}
)

# Set authentication
TOOL_MANAGER.set_http_token("weather_api", "your-api-key")

# Use the HTTP tool
http_tool = TOOL_MANAGER.get_http_tool("weather_api")
success, data = http_tool.get("/forecast", {"city": "New York"})
```

### Vector Database Configuration

```python
from tools import TOOL_MANAGER

# Create a vector database
TOOL_MANAGER.create_vector_db("knowledge_base", "all-MiniLM-L6-v2")

# Add documents to the vector database
vector_db = TOOL_MANAGER.get_vector_db("knowledge_base")
doc_id = vector_db.add_text(
    "Climate change is a global challenge requiring urgent action.",
    {"source": "climate_report", "date": "2024-03-20"}
)

# Search the vector database
results = vector_db.search("global warming solutions", k=5)
```

## Tool Integration Architecture

The tools system is designed with the following components:

1. **Individual Tool Modules**: Standalone wrappers for specific functionality (SQL, HTTP, Vector)
2. **Tool Manager**: Central registry for all tool instances
3. **Tool Registry**: Maps tool IDs to handler functions
4. **Agent Integration**: Extends agents to invoke tools and handle tool responses

This architecture allows for easy addition of new tool types in the future.

## Adding New Tool Types

To add a new tool type:

1. Create a new tool module (e.g., `my_tool.py`)
2. Add factory functions to create tool instances
3. Register tool handler functions in the `tools.py` file
4. Update the Tool Manager to handle the new tool type

## Security Considerations

- Tools provide agents with access to external systems and data
- Consider implementing additional validation and access controls
- Be cautious with database write operations and API POST/PUT requests
- Set appropriate rate limits for HTTP requests
- Review and validate SQL queries before execution in production environments

## Dependencies

The tools extension requires these additional libraries:
- `requests`: For HTTP/API tools
- `pandas`: For data processing
- `faiss-cpu`: For vector embeddings and search
- `sentence-transformers`: For text embedding generation
