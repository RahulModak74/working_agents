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


class ToolManager:
    """Manager class for all agent tools"""
    
    def __init__(self):
        self.sql_tools: Dict[str, SQLTool] = {}
        self.http_tools: Dict[str, HTTPTool] = {}
        self.vector_dbs: Dict[str, VectorDB] = {}
        self.tool_configs: Dict[str, Dict[str, Any]] = self._load_configs()
    
    def _load_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load tool configurations from config file"""
        config_path = os.path.join(TOOLS_DIR, "tool_configs.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("Warning: Invalid tool configuration file")
        return {}
    
    def _save_configs(self):
        """Save tool configurations to file"""
        config_path = os.path.join(TOOLS_DIR, "tool_configs.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.tool_configs, f, indent=2)
    
    def get_sql_tool(self, name: str = "default") -> SQLTool:
        """Get or create an SQL tool instance"""
        if name in self.sql_tools:
            return self.sql_tools[name]
        
        # Create a new SQL tool
        config = self.tool_configs.get("sql", {}).get(name, {})
        db_path = config.get("db_path")
        
        sql_tool = get_sql_tool(db_path)
        self.sql_tools[name] = sql_tool
        
        return sql_tool
    
    def create_sql_tool(self, name: str, db_path: str) -> SQLTool:
        """Create a named SQL tool"""
        sql_tool = get_sql_tool(db_path)
        self.sql_tools[name] = sql_tool
        
        # Save configuration
        if "sql" not in self.tool_configs:
            self.tool_configs["sql"] = {}
        
        self.tool_configs["sql"][name] = {
            "db_path": db_path
        }
        self._save_configs()
        
        return sql_tool
    
    def get_http_tool(self, name: str = "default") -> HTTPTool:
        """Get or create an HTTP tool instance"""
        if name in self.http_tools:
            return self.http_tools[name]
        
        # Create a new HTTP tool
        config = self.tool_configs.get("http", {}).get(name, {})
        base_url = config.get("base_url")
        headers = config.get("headers", {})
        
        http_tool = get_http_tool(base_url, headers)
        
        # Set additional config if available
        if "auth" in config:
            http_tool.set_auth(config["auth"]["username"], config["auth"]["password"])
        
        if "bearer_token" in config:
            http_tool.set_bearer_token(config["bearer_token"])
        
        if "rate_limit" in config:
            http_tool.set_rate_limit(config["rate_limit"])
        
        self.http_tools[name] = http_tool
        return http_tool
    
    def create_http_tool(self, name: str, base_url: Optional[str] = None, 
                         headers: Optional[Dict[str, str]] = None) -> HTTPTool:
        """Create a named HTTP tool"""
        http_tool = get_http_tool(base_url, headers)
        self.http_tools[name] = http_tool
        
        # Save configuration
        if "http" not in self.tool_configs:
            self.tool_configs["http"] = {}
        
        self.tool_configs["http"][name] = {
            "base_url": base_url,
            "headers": headers or {}
        }
        self._save_configs()
        
        return http_tool
    
    def set_http_auth(self, name: str, username: str, password: str):
        """Set authentication for an HTTP tool"""
        http_tool = self.get_http_tool(name)
        http_tool.set_auth(username, password)
        
        # Update configuration
        if "http" not in self.tool_configs:
            self.tool_configs["http"] = {}
        
        if name not in self.tool_configs["http"]:
            self.tool_configs["http"][name] = {}
        
        self.tool_configs["http"][name]["auth"] = {
            "username": username,
            "password": password
        }
        self._save_configs()
    
    def set_http_token(self, name: str, token: str):
        """Set bearer token for an HTTP tool"""
        http_tool = self.get_http_tool(name)
        http_tool.set_bearer_token(token)
        
        # Update configuration
        if "http" not in self.tool_configs:
            self.tool_configs["http"] = {}
        
        if name not in self.tool_configs["http"]:
            self.tool_configs["http"][name] = {}
        
        self.tool_configs["http"][name]["bearer_token"] = token
        self._save_configs()
    
    def get_vector_db(self, name: str = "default") -> VectorDB:
        """Get or create a vector database instance"""
        if name in self.vector_dbs:
            return self.vector_dbs[name]
        
        # Check if this database exists on disk
        db_path = os.path.join(VECTOR_STORAGE, name)
        metadata_path = os.path.join(db_path, "metadata.pkl")
        
        if os.path.exists(metadata_path):
            # Load existing database
            vector_db = VectorDB.load(db_path)
        else:
            # Create a new database
            config = self.tool_configs.get("vector_db", {}).get(name, {})
            model_name = config.get("model_name", "all-MiniLM-L6-v2")
            os.makedirs(db_path, exist_ok=True)
            vector_db = get_vector_db(model_name, db_path)
        
        self.vector_dbs[name] = vector_db
        return vector_db
    
    def create_vector_db(self, name: str, model_name: str = "all-MiniLM-L6-v2") -> VectorDB:
        """Create a named vector database"""
        db_path = os.path.join(VECTOR_STORAGE, name)
        os.makedirs(db_path, exist_ok=True)
        
        vector_db = get_vector_db(model_name, db_path)
        self.vector_dbs[name] = vector_db
        
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
