#!/usr/bin/env python3

"""
SQLite Adapter Module - Simplified version that directly uses SQLite
"""

import os
import sys
import json
import sqlite3
import re
from typing import Any, Dict, List

# Get current directory for absolute paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def extract_database_path(kwargs):
    """Extract database path from kwargs with multiple fallbacks"""
    # Initialize with None
    database = None
    
    # Method 0: Direct string parameter
    if isinstance(kwargs, str):
        database = kwargs
    
    # Method 1: Direct parameter lookup
    if not database and "database" in kwargs:
        database = kwargs["database"]
    
    # Method 2: Find most recently created SQLite database in current directory
    if not database:
        try:
            import glob
            import os
            
            # Find all .sqlite and .db files
            sqlite_files = glob.glob('*.sqlite') + glob.glob('*.db')
            
            if sqlite_files:
                # Sort by modification time, most recent first
                sqlite_files.sort(key=os.path.getmtime, reverse=True)
                database = sqlite_files[0]
                print(f"Found most recently created database: {database}")
        except Exception as e:
            print(f"Error finding database file: {e}")
    
    # Method 3: Content text analysis
    if not database:
        content = kwargs.get("content", "")
        if isinstance(content, str):
            # Try to find database name
            db_patterns = [
                r"database ['\"]?(.*?)['\"]?",
                r"database: ['\"]?(.*?)['\"]?",
                r"database file ['\"]?(.*?)['\"]?"
            ]
            
            for pattern in db_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Clean up quotes if present
                    if (candidate.startswith("'") and candidate.endswith("'")) or \
                       (candidate.startswith('"') and candidate.endswith('"')):
                        candidate = candidate[1:-1]
                    
                    # Only use if it looks like a database name
                    if candidate and ";" not in candidate and " " not in candidate:
                        database = candidate
                        break
    
    # Method 4: Nested structure
    if not database:
        for k, v in kwargs.items():
            if isinstance(v, dict) and "database" in v:
                database = v["database"]
                break
    
    # Fallback to default or most recent
    if not database:
        # Try to find any .sqlite or .db file
        import glob
        sqlite_files = glob.glob('*.sqlite') + glob.glob('*.db')
        
        if sqlite_files:
            database = sqlite_files[0]
            print(f"Defaulting to: {database}")
        else:
            # Last resort
            database = "test_db.sqlite"
            print(f"WARNING: No database specified, defaulting to {database}")
    
    # Convert to absolute path if relative
    if not os.path.isabs(database):
        # Prioritize current working directory
        current_dir = os.getcwd()
        database_path = os.path.join(current_dir, database)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(database_path), exist_ok=True)
    else:
        database_path = database
    
    print(f"Using full database path: {database_path}")
    return database_path
def execute_sqlite_query(database_path, query):
    """Execute a SQLite query and return results"""
    try:
        print(f"Connecting to SQLite database: {database_path}")
        
        # Ensure the directory exists
        db_dir = os.path.dirname(database_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        # Connect to database (this creates it if it doesn't exist)
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Execute the query
        try:
            cursor.execute(query)
            conn.commit()
            
            # Check if this is a SELECT query by seeing if it returns rows
            if cursor.description:
                # Convert rows to dictionaries
                columns = [col[0] for col in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # For readability in the results, limit to 10 rows if many
                display_rows = rows[:10]
                has_more = len(rows) > 10
                
                result = {
                    "success": True,
                    "query_type": "SELECT",
                    "row_count": len(rows),
                    "results": display_rows
                }
                
                if has_more:
                    result["note"] = f"Showing first 10 of {len(rows)} rows"
            else:
                # This was likely an INSERT, UPDATE, DELETE or CREATE
                result = {
                    "success": True,
                    "query_type": "DML/DDL",
                    "rows_affected": cursor.rowcount,
                }
                
            return result
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"Exception in sqlite_query: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def get_sqlite_tables(database_path):
    """Get list of tables in SQLite database"""
    try:
        print(f"Getting tables from: {database_path}")
        
        # Create database file if it doesn't exist
        if not os.path.exists(database_path):
            print(f"Database file does not exist, creating: {database_path}")
            conn = sqlite3.connect(database_path)
            conn.close()
        
        # Connect and get table list
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Query for all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "tables": tables,
            "database_path": database_path
        }
    except Exception as e:
        print(f"Error getting SQLite tables: {e}")
        return {
            "success": False, 
            "error": str(e),
            "database_path": database_path
        }

def get_sqlite_schema(database_path, table_name):
    """Get schema for a table in SQLite database"""
    try:
        print(f"Getting schema for table {table_name} from: {database_path}")
        
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        
        # Query for table info
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        # Format column info
        formatted_columns = []
        for col in columns:
            formatted_columns.append({
                "name": col[1],
                "type": col[2],
                "notnull": bool(col[3]),
                "default_value": col[4],
                "is_primary_key": bool(col[5])
            })
        
        cursor.close()
        conn.close()
        
        return {
            "success": True,
            "table_name": table_name,
            "columns": formatted_columns,
            "database_path": database_path
        }
    except Exception as e:
        print(f"Error getting SQLite schema: {e}")
        return {
            "success": False,
            "error": str(e),
            "table_name": table_name,
            "database_path": database_path
        }

def extract_table_name(kwargs):
    """Extract table name from kwargs with multiple fallbacks"""
    # Initialize with None
    table = None
    
    # Method 1: Direct parameter lookup
    if "table" in kwargs:
        table = kwargs["table"]
    
    # Method 2: Content text analysis
    if not table:
        content = kwargs.get("content", "")
        if isinstance(content, str):
            # Look for exact table references
            table_patterns = [
                r"table ['\"]?(.*?)['\"]?",
                r"table: ['\"]?(.*?)['\"]?",
                r"table name ['\"]?(.*?)['\"]?"
            ]
            
            for pattern in table_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    candidate = match.group(1).strip()
                    # Clean up quotes
                    if (candidate.startswith("'") and candidate.endswith("'")) or \
                       (candidate.startswith('"') and candidate.endswith('"')):
                        candidate = candidate[1:-1]
                    
                    # Only use if it looks like a table name
                    if candidate and ";" not in candidate:
                        table = candidate
                        break
    
    # Method 3: Nested structure
    if not table:
        for k, v in kwargs.items():
            if isinstance(v, dict) and "table" in v:
                table = v["table"]
                break
    
    return table

def extract_query(kwargs):
    """Extract SQL query from kwargs with multiple fallbacks"""
    # Initialize with None
    query = None
    
    # Method 1: Direct parameter lookup
    if "query" in kwargs:
        query = kwargs["query"]
    
    # Method 2: Content text analysis
    if not query:
        content = kwargs.get("content", "")
        if isinstance(content, str):
            # Look for common SQL statements
            sql_patterns = [
                r"query: ['\"]([^;]+;)['\"]",
                r"query ['\"]([^;]+;)['\"]",
                r"following query: ['\"]([^;]+;)['\"]",
                r"(CREATE TABLE [^;]+;)",
                r"(INSERT INTO [^;]+;)",
                r"(SELECT [^;]+;)",
                r"(UPDATE [^;]+;)",
                r"(DELETE FROM [^;]+;)"
            ]
            
            for pattern in sql_patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    query = match.group(1).strip()
                    break
    
    # Method 3: Nested structure
    if not query:
        for k, v in kwargs.items():
            if isinstance(v, dict) and "query" in v:
                query = v["query"]
                break
    
    return query

# Adapter functions
def sql_query(**kwargs):
    """Adapter for SQL query execution"""
    print(f"\n--- SQL QUERY EXECUTION ---")
    print(f"Received parameters: {list(kwargs.keys())}")
    
    # Extract the database path and query
    database_path = extract_database_path(kwargs)
    query = extract_query(kwargs)
    
    # Validate we have required parameters
    if not query:
        error_msg = "SQL query not provided"
        print(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}
    
    print(f"Executing query: {query}")
    
    # Execute the query
    return execute_sqlite_query(database_path, query)

def sql_tables(**kwargs):
    """Adapter for SQL tables listing"""
    print(f"\n--- SQL TABLES LISTING ---")
    print(f"Received parameters: {list(kwargs.keys())}")
    
    # Extract the database path
    database_path = extract_database_path(kwargs)
    
    # Get the tables
    return get_sqlite_tables(database_path)

def sql_schema(**kwargs):
    """Adapter for SQL schema retrieval"""
    print(f"\n--- SQL SCHEMA RETRIEVAL ---")
    print(f"Received parameters: {list(kwargs.keys())}")
    
    # Extract the database path and table name
    database_path = extract_database_path(kwargs)
    table_name = extract_table_name(kwargs)
    
    # Validate we have required parameters
    if not table_name:
        error_msg = "Table name not provided"
        print(f"ERROR: {error_msg}")
        return {"success": False, "error": error_msg}
    
    # Get the schema
    return get_sqlite_schema(database_path, table_name)

# Tool registry
TOOL_REGISTRY = {
    "sql:query": sql_query,
    "sql:tables": sql_tables,
    "sql:schema": sql_schema
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
            import traceback
            traceback.print_exc()
            return {"error": f"Tool execution failed: {str(e)}"}
    else:
        print(f"Unknown tool: {tool_id}")
        return {"error": f"Unknown tool: {tool_id}"}

# Allow direct execution for testing
if __name__ == "__main__":
    # Example usage
    result = execute_tool("sql:tables", database="test_sqlite.db")
    print(result)
