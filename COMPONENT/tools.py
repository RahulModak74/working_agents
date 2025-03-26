#!/usr/bin/env python3

import os
import json
from typing import Dict, Any, Optional, List, Union

from sql_tool import SQLTool, get_sql_tool
from http_tool import HTTPTool, get_http_tool
from vector_db import VectorDB, get_vector_db
from config import CONFIG

# Create a tools directory
TOOLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_data")
os.makedirs(TOOLS_DIR, exist_ok=True)

# Vector DB storage
VECTOR_STORAGE = os.path.join(TOOLS_DIR, "vector_storage")
os.makedirs(VECTOR_STORAGE, exist_ok=True)

# Define tool registry
TOOL_REGISTRY = {}


db
        
        # Save configuration
        if "vector_db" not in self.tool_configs:
            self.tool_configs["vector_db"] = {}
        
        self.tool_configs["vector_db"][name] = {
            "model_name": model_name,
            "path": db_path
        }
        self._save_configs()
        
        return vector_db
    
    def save_all_vector_dbs(self):
        """Save all vector databases to disk"""
        for name, vector_db in self.vector_dbs.items():
            db_path = os.path.join(VECTOR_STORAGE, name)
            vector_db.save(db_path)
    
    def get_available_tools(self) -> Dict[str, List[str]]:
        """Get information about all available tools"""
        return {
            "sql": list(self.sql_tools.keys()),
            "http": list(self.http_tools.keys()),
            "vector_db": list(self.vector_dbs.keys())
        }


# Create a global tool manager instance
TOOL_MANAGER = ToolManager()


def register_tool(tool_type: str, name: str, handler_fn: callable):
    """Register a tool function in the global registry"""
    TOOL_REGISTRY[f"{tool_type}:{name}"] = handler_fn


# Register SQL tools
def sql_query(name: str = "default", query: str = "", params: tuple = None) -> Dict[str, Any]:
    """Execute an SQL query and return results"""
    sql_tool = TOOL_MANAGER.get_sql_tool(name)
    success, results = sql_tool.execute_query(query, params)
    return {
        "success": success,
        "results": results
    }

register_tool("sql", "query", sql_query)


def sql_tables(name: str = "default") -> List[str]:
    """Get list of tables in the database"""
    sql_tool = TOOL_MANAGER.get_sql_tool(name)
    return sql_tool.get_tables()

register_tool("sql", "tables", sql_tables)


def sql_schema(name: str = "default", table: str = None) -> Dict[str, Any]:
    """Get schema for tables in the database"""
    sql_tool = TOOL_MANAGER.get_sql_tool(name)
    return sql_tool.get_schema(table)

register_tool("sql", "schema", sql_schema)


# Register HTTP tools
def http_get(name: str = "default", endpoint: str = "", params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make an HTTP GET request"""
    http_tool = TOOL_MANAGER.get_http_tool(name)
    success, data = http_tool.get(endpoint, params)
    return {
        "success": success,
        "data": data,
        "status_code": http_tool.get_status_code()
    }

register_tool("http", "get", http_get)


def http_post(name: str = "default", endpoint: str = "", 
              json_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make an HTTP POST request with JSON data"""
    http_tool = TOOL_MANAGER.get_http_tool(name)
    success, data = http_tool.post(endpoint, json_data=json_data)
    return {
        "success": success,
        "data": data,
        "status_code": http_tool.get_status_code()
    }

register_tool("http", "post", http_post)


# Register Vector DB tools
def vector_search(name: str = "default", query: str = "", k: int = 5) -> List[Dict[str, Any]]:
    """Search the vector database for similar documents"""
    vector_db = TOOL_MANAGER.get_vector_db(name)
    return vector_db.search(query, k)

register_tool("vector_db", "search", vector_search)


def vector_add(name: str = "default", text: str = "", metadata: Dict[str, Any] = None) -> str:
    """Add a document to the vector database"""
    vector_db = TOOL_MANAGER.get_vector_db(name)
    doc_id = vector_db.add_text(text, metadata)
    vector_db.save()  # Save after adding
    return doc_id

register_tool("vector_db", "add", vector_add)


def vector_batch_add(name: str = "default", texts: List[str] = None, 
                     metadatas: List[Dict[str, Any]] = None) -> List[str]:
    """Add multiple documents to the vector database"""
    if not texts:
        return []
    
    vector_db = TOOL_MANAGER.get_vector_db(name)
    doc_ids = vector_db.add_texts(texts, metadatas)
    vector_db.save()  # Save after adding
    return doc_ids

register_tool("vector_db", "batch_add", vector_batch_add)


def execute_tool(tool_id: str, **kwargs) -> Any:
    """Execute a tool by its ID with the provided parameters"""
    if tool_id not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_id}"}
    
    try:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    except Exception as e:
        return {"error": str(e)}
