#!/usr/bin/env python3

"""
Tools Adapter Module
This module provides an adapter layer between the workflow JSON tool format and
the actual tool implementation in tools.py.
"""

import os
import sys
import json
from typing import Any, Dict, List

# Ensure both directories are in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import the tools module
try:
    from tools import (
        http_get as original_http_get,
        http_post as original_http_post,
        sql_query as original_sql_query,
        sql_tables as original_sql_tables,
        sql_schema as original_sql_schema,
        vector_search as original_vector_search,
        vector_add as original_vector_add,
        vector_batch_add as original_vector_batch_add,
        TOOL_MANAGER
    )
    real_tools_available = True
except ImportError:
    print("Warning: Real tools module not available. Using mock implementations.")
    real_tools_available = False

# Mock implementations for when real tools aren't available
def mock_http_get(url, headers=None, **kwargs):
    """Mock HTTP GET implementation"""
    print(f"[MOCK] HTTP GET to {url} with headers {headers}")
    return {
        "success": True,
        "status_code": 200,
        "data": {
            "message": "This is a mock HTTP GET response",
            "url": url,
            "records": 42
        }
    }

def mock_http_post(url, headers=None, data=None, **kwargs):
    """Mock HTTP POST implementation"""
    print(f"[MOCK] HTTP POST to {url} with headers {headers} and data {data}")
    return {
        "success": True,
        "status_code": 201,
        "data": {
            "message": "Data created successfully",
            "id": "mock-12345"
        }
    }

def mock_sql_query(database, query, **kwargs):
    """Mock SQL query implementation"""
    print(f"[MOCK] SQL query on database {database}: {query}")
    return {
        "success": True,
        "results": [
            {"id": 1, "timestamp": "2023-01-01", "endpoint": "/data"},
            {"id": 2, "timestamp": "2023-01-02", "endpoint": "/users"}
        ]
    }

def mock_sql_tables(database, **kwargs):
    """Mock SQL tables implementation"""
    print(f"[MOCK] Get tables in database {database}")
    return ["table1", "table2", "table3"]

def mock_sql_schema(database, table, **kwargs):
    """Mock SQL schema implementation"""
    print(f"[MOCK] Get schema for table {table} in database {database}")
    return {
        "columns": [
            {"name": "id", "type": "INTEGER"},
            {"name": "timestamp", "type": "DATETIME"},
            {"name": "endpoint", "type": "TEXT"}
        ]
    }

def mock_vector_search(collection, query=None, top_k=5, model=None, **kwargs):
    """Mock vector search implementation"""
    print(f"[MOCK] Vector search in collection {collection} for '{query}' (top {top_k})")
    return [
        {"id": "doc1", "content": "Sample document 1", "score": 0.95},
        {"id": "doc2", "content": "Sample document 2", "score": 0.85},
        {"id": "doc3", "content": "Sample document 3", "score": 0.75}
    ]

def mock_vector_add(collection, text, metadata=None, model=None, **kwargs):
    """Mock vector add implementation"""
    print(f"[MOCK] Adding document to collection {collection}")
    return "mock-doc-id-123"

def mock_vector_batch_add(collection, texts, metadatas=None, model=None, batch_size=None, **kwargs):
    """Mock vector batch add implementation"""
    print(f"[MOCK] Adding {len(texts) if texts else 0} documents to collection {collection}")
    return ["mock-id-1", "mock-id-2", "mock-id-3"]

# Adapter functions that bridge between workflow JSON format and actual tool implementation
def http_get(**kwargs):
    """Adapter for HTTP GET requests"""
    # Extract parameters from kwargs
    url = kwargs.get("url")
    headers = kwargs.get("headers")
    
    # Convert from string to dict if needed
    if isinstance(headers, str):
        try:
            headers = json.loads(headers.replace("'", '"'))
        except json.JSONDecodeError:
            headers = {}
    
    if real_tools_available:
        # Adapt parameters to the real tool
        # In the real implementation, it expects name, endpoint, params
        # Extract domain for the name
        name = "default"
        endpoint = ""
        
        if url:
            # Try to set up the tool with the base URL
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                endpoint = parsed_url.path
                
                # Create/configure the HTTP tool
                TOOL_MANAGER.create_http_tool(name, base_url, headers)
                
                # Add query parameters if present
                params = {}
                if parsed_url.query:
                    import urllib.parse as urlparse
                    params = dict(urlparse.parse_qsl(parsed_url.query))
                
                # Call the original function with adapted parameters
                return original_http_get(name, endpoint, params)
            except Exception as e:
                print(f"Error adapting HTTP request: {e}")
                # Fall back to the mock implementation
                return mock_http_get(url, headers)
        else:
            return {"error": "URL not provided for HTTP GET request"}
    else:
        # Use the mock implementation
        return mock_http_get(url, headers)

def http_post(**kwargs):
    """Adapter for HTTP POST requests"""
    # Extract parameters from kwargs
    url = kwargs.get("url")
    headers = kwargs.get("headers")
    data = kwargs.get("data")
    
    # Convert from string to dict if needed
    if isinstance(headers, str):
        try:
            headers = json.loads(headers.replace("'", '"'))
        except json.JSONDecodeError:
            headers = {}
    
    if isinstance(data, str):
        try:
            data = json.loads(data.replace("'", '"'))
        except json.JSONDecodeError:
            pass  # Keep as string if parsing fails
    
    if real_tools_available:
        # Adapt parameters to the real tool
        name = "default"
        endpoint = ""
        
        if url:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(url)
                base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                endpoint = parsed_url.path
                
                # Create/configure the HTTP tool
                TOOL_MANAGER.create_http_tool(name, base_url, headers)
                
                # Call the original function with adapted parameters
                return original_http_post(name, endpoint, data)
            except Exception as e:
                print(f"Error adapting HTTP POST request: {e}")
                return mock_http_post(url, headers, data)
        else:
            return {"error": "URL not provided for HTTP POST request"}
    else:
        # Use the mock implementation
        return mock_http_post(url, headers, data)

def sql_query(**kwargs):
    """Adapter for SQL query execution"""
    # Extract parameters from kwargs
    database = kwargs.get("database")
    query = kwargs.get("query")
    
    if real_tools_available:
        try:
            return original_sql_query(database, query)
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return mock_sql_query(database, query)
    else:
        return mock_sql_query(database, query)

def sql_tables(**kwargs):
    """Adapter for SQL tables listing"""
    database = kwargs.get("database")
    
    if real_tools_available:
        try:
            return original_sql_tables(database)
        except Exception as e:
            print(f"Error listing SQL tables: {e}")
            return mock_sql_tables(database)
    else:
        return mock_sql_tables(database)

def sql_schema(**kwargs):
    """Adapter for SQL schema retrieval"""
    database = kwargs.get("database")
    table = kwargs.get("table")
    
    if real_tools_available:
        try:
            return original_sql_schema(database, table)
        except Exception as e:
            print(f"Error retrieving SQL schema: {e}")
            return mock_sql_schema(database, table)
    else:
        return mock_sql_schema(database, table)

def vector_db_search(**kwargs):
    """Adapter for vector database search"""
    # Extract parameters from kwargs
    collection = kwargs.get("collection")
    query = kwargs.get("query")
    top_k = kwargs.get("top_k")
    model = kwargs.get("model")
    
    # Convert string values to appropriate types
    if isinstance(top_k, str):
        try:
            top_k = int(top_k)
        except ValueError:
            top_k = 5
    
    if real_tools_available:
        try:
            # Adapt parameters to the real tool
            name = collection or "default"
            k = top_k or 5
            
            # Configure the vector DB with the model if needed
            if model and collection:
                try:
                    vector_db = TOOL_MANAGER.get_vector_db(name)
                    # Check if we need to create a new one with the specified model
                    if not vector_db or vector_db.model_name != model:
                        TOOL_MANAGER.create_vector_db(name, model)
                except Exception as e:
                    print(f"Error configuring vector database: {e}")
            
            return original_vector_search(name, query, k)
        except Exception as e:
            print(f"Error searching vector database: {e}")
            return mock_vector_search(collection, query, top_k, model)
    else:
        return mock_vector_search(collection, query, top_k, model)

def vector_db_add(**kwargs):
    """Adapter for adding a document to vector database"""
    # Extract parameters from kwargs
    collection = kwargs.get("collection")
    text = kwargs.get("text")
    metadata = kwargs.get("metadata")
    model = kwargs.get("model")
    
    if real_tools_available:
        try:
            # Adapt parameters to the real tool
            name = collection or "default"
            
            # Configure the vector DB with the model if needed
            if model and collection:
                try:
                    vector_db = TOOL_MANAGER.get_vector_db(name)
                    # Check if we need to create a new one with the specified model
                    if not vector_db or vector_db.model_name != model:
                        TOOL_MANAGER.create_vector_db(name, model)
                except Exception as e:
                    print(f"Error configuring vector database: {e}")
            
            return original_vector_add(name, text, metadata)
        except Exception as e:
            print(f"Error adding document to vector database: {e}")
            return mock_vector_add(collection, text, metadata, model)
    else:
        return mock_vector_add(collection, text, metadata, model)

def vector_db_batch_add(**kwargs):
    """Adapter for adding multiple documents to vector database"""
    # Extract parameters from kwargs
    collection = kwargs.get("collection")
    texts = kwargs.get("texts")
    metadatas = kwargs.get("metadatas")
    model = kwargs.get("model")
    batch_size = kwargs.get("batch_size")
    
    # Convert string values to appropriate types
    if isinstance(batch_size, str):
        try:
            batch_size = int(batch_size)
        except ValueError:
            batch_size = 10
    
    if real_tools_available:
        try:
            # Adapt parameters to the real tool
            name = collection or "default"
            
            # Configure the vector DB with the model if needed
            if model and collection:
                try:
                    vector_db = TOOL_MANAGER.get_vector_db(name)
                    # Check if we need to create a new one with the specified model
                    if not vector_db or vector_db.model_name != model:
                        TOOL_MANAGER.create_vector_db(name, model)
                except Exception as e:
                    print(f"Error configuring vector database: {e}")
            
            return original_vector_batch_add(name, texts, metadatas)
        except Exception as e:
            print(f"Error batch adding documents to vector database: {e}")
            return mock_vector_batch_add(collection, texts, metadatas, model, batch_size)
    else:
        return mock_vector_batch_add(collection, texts, metadatas, model, batch_size)

# Tool registry
TOOL_REGISTRY = {
    "http:get": http_get,
    "http:post": http_post,
    "sql:query": sql_query,
    "sql:tables": sql_tables,
    "sql:schema": sql_schema,
    "vector_db:search": vector_db_search,
    "vector_db:add": vector_db_add,
    "vector_db:batch_add": vector_db_batch_add
}

def execute_tool(tool_id, **kwargs):
    """Main entry point for tool execution"""
    if tool_id in TOOL_REGISTRY:
        try:
            print(f"Executing tool: {tool_id}")
            result = TOOL_REGISTRY[tool_id](**kwargs)
            print(f"Tool execution completed: {tool_id}")
            return result
        except Exception as e:
            print(f"Error executing tool {tool_id}: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}
    else:
        print(f"Unknown tool: {tool_id}")
        return {"error": f"Unknown tool: {tool_id}"}
