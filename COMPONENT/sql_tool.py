#!/usr/bin/env python3

import sqlite3
import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple


class SQLTool:
    """Tool for executing SQL queries against various database types"""
    
    def __init__(self, db_path: str = None):
        """Initialize with optional default database path"""
        self.db_path = db_path
        self.connection = None
        self.cursor = None
    
    def connect(self, db_path: str = None) -> bool:
        """Connect to a SQLite database"""
        try:
            db_path = db_path or self.db_path
            if not db_path:
                raise ValueError("Database path not provided")
            
            self.connection = sqlite3.connect(db_path)
            self.connection.row_factory = sqlite3.Row  # Return rows as dictionaries
            self.cursor = self.connection.cursor()
            self.db_path = db_path
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close the current database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None
    
    def execute_query(self, query: str, params: tuple = None) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """Execute a SQL query and return results as a list of dictionaries"""
        try:
            if not self.connection:
                if not self.connect():
                    return False, "Not connected to a database"
            
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            
            # Check if this is a SELECT query
            if query.strip().upper().startswith("SELECT"):
                columns = [desc[0] for desc in self.cursor.description]
                results = []
                for row in self.cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                return True, results
            else:
                self.connection.commit()
                return True, f"Query executed successfully. Rows affected: {self.cursor.rowcount}"
        except Exception as e:
            return False, f"Error executing query: {str(e)}"
    
    def get_tables(self) -> List[str]:
        """Get a list of tables in the current database"""
        if not self.connection:
            if not self.connect():
                return []
        
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in self.cursor.fetchall()]
    
    def get_schema(self, table_name: str = None) -> Dict[str, Any]:
        """Get the schema for the specified table or all tables"""
        if not self.connection:
            if not self.connect():
                return {}
        
        schema_info = {}
        
        if table_name:
            tables = [table_name]
        else:
            tables = self.get_tables()
        
        for table in tables:
            self.cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for col in self.cursor.fetchall():
                columns.append({
                    "name": col["name"],
                    "type": col["type"],
                    "notnull": bool(col["notnull"]),
                    "default": col["dflt_value"],
                    "pk": bool(col["pk"])
                })
            schema_info[table] = columns
        
        return schema_info
    
    def load_sql_file(self, file_path: str) -> Tuple[bool, str]:
        """Execute SQL statements from a file"""
        try:
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            with open(file_path, 'r') as f:
                sql_script = f.read()
            
            if not self.connection:
                if not self.connect():
                    return False, "Not connected to a database"
            
            self.cursor.executescript(sql_script)
            self.connection.commit()
            return True, "SQL script executed successfully"
        except Exception as e:
            return False, f"Error executing SQL script: {str(e)}"
    
    def export_to_json(self, query: str, output_file: str) -> Tuple[bool, str]:
        """Export query results to a JSON file"""
        success, results = self.execute_query(query)
        if not success:
            return False, results
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            return True, f"Results exported to {output_file}"
        except Exception as e:
            return False, f"Error exporting to JSON: {str(e)}"
    
    def export_to_csv(self, query: str, output_file: str) -> Tuple[bool, str]:
        """Export query results to a CSV file"""
        success, results = self.execute_query(query)
        if not success:
            return False, results
        
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            return True, f"Results exported to {output_file}"
        except Exception as e:
            return False, f"Error exporting to CSV: {str(e)}"
    
    def import_from_csv(self, file_path: str, table_name: str, if_exists: str = 'append') -> Tuple[bool, str]:
        """Import data from a CSV file into a database table"""
        try:
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            if not self.connection:
                if not self.connect():
                    return False, "Not connected to a database"
            
            df = pd.read_csv(file_path)
            conn = sqlite3.connect(self.db_path)
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
            conn.close()
            
            return True, f"Data imported to table {table_name}"
        except Exception as e:
            return False, f"Error importing from CSV: {str(e)}"


def get_sql_tool(db_path: Optional[str] = None) -> SQLTool:
    """Factory function to get a SQL tool instance"""
    return SQLTool(db_path)
