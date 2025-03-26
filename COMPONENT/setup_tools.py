def get_http_tool(base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> HTTPTool:
    """Factory function to get an HTTP tool instance"""
    return HTTPTool(base_url, headers)""",

        "vector_db.py": """#!/usr/bin/env python3

import os
import json
import numpy as np
import faiss
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer


class VectorDB:
    """Vector database for semantic search using FAISS and sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = None,
                 storage_dir: str = "./vector_storage"):
        """Initialize the vector database with the specified model"""
        self.model_name = model_name
        self.storage_dir = storage_dir
        self.encoder = None
        self.index = None
        self.dimension = dimension
        self.metadata = {}  # Maps ID to metadata
        self.id_map = {}    # Maps FAISS internal ID to our external ID
        self.rev_id_map = {}  # Maps our external ID to FAISS internal ID
        self.next_id = 0
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def _load_encoder(self):
        """Load the sentence transformer model"""
        if self.encoder is None:
            try:
                self.encoder =#!/usr/bin/env python3

import os
import subprocess
import sys
import shutil
import re
import pathlib
import importlib.util

REQUIRED_PACKAGES = [
    "jsonschema>=4.0.0",
    "pandas>=1.3.0",
    "requests>=2.26.0",
    "faiss-cpu>=1.7.2",
    "sentence-transformers>=2.2.0",
    "numpy>=1.21.0"
]

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        for package in REQUIRED_PACKAGES:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install some dependencies")
        print("Please manually run: pip install jsonschema pandas requests faiss-cpu sentence-transformers numpy")


def check_file_exists(filename):
    """Check if a file exists"""
    return os.path.exists(filename)


def backup_file(filename):
    """Create a backup of a file"""
    if check_file_exists(filename):
        backup_name = f"{filename}.bak"
        shutil.copy2(filename, backup_name)
        print(f"✅ Created backup: {backup_name}")
        return True
    return False


def insert_into_file(filename, search_marker, new_code):
    """Insert code into an existing file at a specific marker location"""
    if not check_file_exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the code appears to be already inserted
    if new_code.strip() in content:
        print(f"⚠️ Code already appears to be in {filename}")
        return True
    
    # Find the insertion point
    if search_marker in content:
        # Insert after the marker
        modified_content = content.replace(search_marker, search_marker + "\n" + new_code)
        
        # Create backup first
        backup_file(filename)
        
        # Write the modified content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ Modified {filename}")
        return True
    else:
        print(f"❌ Could not find insertion point in {filename}")
        return False


def get_class_definition_end(content, class_name):
    """Find the end of a class definition in file content"""
    # This is a simplified approach - for a robust solution, we would need a proper Python parser
    class_pattern = rf"class\s+{class_name}\s*\("
    class_match = re.search(class_pattern, content)
    
    if not class_match:
        return -1
    
    # Find the end of the class by counting indentation
    start_pos = class_match.start()
    lines = content[start_pos:].split('\n')
    
    # Get the indentation of the first line after class definition
    first_line = lines[1]
    class_indent = len(first_line) - len(first_line.lstrip())
    
    # Find where the indentation level returns to less than or equal to the class declaration
    line_count = 1
    for line in lines[2:]:
        line_count += 1
        
        # Skip empty lines
        if not line.strip():
            continue
        
        # Check indentation level
        indent = len(line) - len(line.lstrip())
        if indent <= class_indent and line.strip():
            return start_pos + sum(len(l) + 1 for l in lines[:line_count-1])
    
    # If we can't find the end, return the end of the file
    return len(content)


def insert_methods_into_class(filename, class_name, new_methods):
    """Insert methods into an existing class"""
    if not check_file_exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the code appears to be already inserted
    if new_methods.strip() in content:
        print(f"⚠️ Methods already appear to be in {class_name}")
        return True
    
    # Find the end of the class
    end_pos = get_class_definition_end(content, class_name)
    
    if end_pos > 0:
        # Insert before the end of the class
        modified_content = content[:end_pos] + "\n" + new_methods + "\n" + content[end_pos:]
        
        # Create backup first
        backup_file(filename)
        
        # Write the modified content
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"✅ Added methods to {class_name} in {filename}")
        return True
    else:
        print(f"❌ Could not find class {class_name} in {filename}")
        return False


def add_import_statement(filename, import_statement):
    """Add an import statement to a file if not already present"""
    if not check_file_exists(filename):
        print(f"❌ File not found: {filename}")
        return False
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if the import already exists
    if import_statement in content:
        return True
    
    # Find the last import statement
    import_pattern = r'^import\s+.*|^from\s+.*\s+import\s+.*'
    matches = list(re.finditer(import_pattern, content, re.MULTILINE))
    
    if matches:
        last_import = matches[-1]
        insert_position = last_import.end()
        
        # Insert after the last import
        modified_content = content[:insert_position] + "\n" + import_statement + content[insert_position:]
    else:
        # No imports found, add at the beginning after any comments/shebang
        first_code_line = 0
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                first_code_line = i
                break
        
        # Join the content back with the import added
        modified_content = '\n'.join(lines[:first_code_line]) + "\n" + import_statement + "\n" + '\n'.join(lines[first_code_line:])
    
    # Create backup first
    backup_file(filename)
    
    # Write the modified content
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Added import to {filename}")
    return True


def ensure_directories():
    """Create necessary directories"""
    dirs = ["./tools_data", "./tools_data/vector_storage"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✅ Ensured directory: {d}")


def install_tool_files():
    """Copy tool files to the current directory"""
    tool_files = {
        "sql_tool.py": """#!/usr/bin/env python3

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
    return SQLTool(db_path)""",
        
        "http_tool.py": """#!/usr/bin/env python3

import requests
import json
import os
import time
from typing import Dict, Any, Optional, Union, Tuple, List
from urllib.parse import urlparse, urljoin


class HTTPTool:
    """Tool for making HTTP requests to REST APIs"""
    
    def __init__(self, base_url: str = None, headers: Dict[str, str] = None, 
                 auth: Tuple[str, str] = None, timeout: int = 30,
                 verify_ssl: bool = True):
        """Initialize with optional base URL and default headers"""
        self.base_url = base_url
        self.headers = headers or {}
        self.default_timeout = timeout
        self.auth = auth
        self.verify_ssl = verify_ssl
        self.last_response = None
        self.rate_limit_delay = 0  # Seconds between requests
        self.last_request_time = 0
    
    def _prepare_url(self, endpoint: str) -> str:
        """Prepare the full URL from the endpoint"""
        if not endpoint.startswith(('http://', 'https://')):
            if not self.base_url:
                raise ValueError("Base URL not set and endpoint is not a full URL")
            return urljoin(self.base_url, endpoint)
        return endpoint
    
    def _respect_rate_limit(self):
        """Respect rate limiting by waiting if needed"""
        if self.rate_limit_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
    
    def set_base_url(self, base_url: str):
        """Set or update the base URL"""
        self.base_url = base_url
    
    def set_headers(self, headers: Dict[str, str]):
        """Set or update the default headers"""
        self.headers = headers
    
    def add_header(self, key: str, value: str):
        """Add or update a single header"""
        self.headers[key] = value
    
    def set_auth(self, username: str, password: str):
        """Set basic authentication credentials"""
        self.auth = (username, password)
    
    def set_bearer_token(self, token: str):
        """Set bearer token authentication"""
        self.headers['Authorization'] = f"Bearer {token}"
    
    def set_rate_limit(self, requests_per_minute: int):
        """Set rate limiting for requests"""
        if requests_per_minute <= 0:
            self.rate_limit_delay = 0
        else:
            self.rate_limit_delay = 60.0 / requests_per_minute
    
    def get(self, endpoint: str, params: Dict[str, Any] = None, 
            headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a GET request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.get(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def post(self, endpoint: str, data: Dict[str, Any] = None, 
             json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
             headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a POST request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.post(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def put(self, endpoint: str, data: Dict[str, Any] = None, 
            json_data: Dict[str, Any] = None, params: Dict[str, Any] = None,
            headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a PUT request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.put(
                url,
                data=data,
                json=json_data,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def delete(self, endpoint: str, params: Dict[str, Any] = None, 
               headers: Dict[str, str] = None, timeout: int = None) -> Tuple[bool, Any]:
        """Make a DELETE request to the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.delete(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=timeout or self.default_timeout,
                verify=self.verify_ssl
            )
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}
    
    def _process_response(self, response) -> Tuple[bool, Any]:
        """Process the response and return data in appropriate format"""
        success = 200 <= response.status_code < 300
        
        # Try to parse as JSON first
        try:
            data = response.json()
        except json.JSONDecodeError:
            # If not JSON, return text
            data = response.text
        
        if not success:
            return False, {
                "status_code": response.status_code,
                "error": data if isinstance(data, dict) else {"message": data},
                "headers": dict(response.headers)
            }
        
        return True, data
    
    def get_response_headers(self) -> Dict[str, str]:
        """Get headers from the last response"""
        if self.last_response:
            return dict(self.last_response.headers)
        return {}
    
    def get_status_code(self) -> Optional[int]:
        """Get status code from the last response"""
        if self.last_response:
            return self.last_response.status_code
        return None
    
    def download_file(self, endpoint: str, output_path: str, params: Dict[str, Any] = None,
                      headers: Dict[str, str] = None) -> Tuple[bool, str]:
        """Download a file from the specified endpoint"""
        try:
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            response = requests.get(
                url,
                params=params,
                headers=merged_headers or None,
                auth=self.auth,
                stream=True,
                verify=self.verify_ssl
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True, f"File downloaded successfully to {output_path}"
            else:
                return False, f"Failed to download file: HTTP {response.status_code}"
        except Exception as e:
            return False, f"Error downloading file: {str(e)}"
    
    def upload_file(self, endpoint: str, file_path: str, form_field: str = 'file',
                    extra_data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Tuple[bool, Any]:
        """Upload a file to the specified endpoint"""
        try:
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            self._respect_rate_limit()
            url = self._prepare_url(endpoint)
            
            files = {form_field: open(file_path, 'rb')}
            data = extra_data or {}
            
            merged_headers = {**self.headers, **(headers or {})}
            self.last_request_time = time.time()
            self.last_response = requests.post(
                url,
                files=files,
                data=data,
                headers=merged_headers or None,
                auth=self.auth,
                timeout=self.default_timeout,
                verify=self.verify_ssl
            )
            
            # Close the file
            for file_obj in files.values():
                file_obj.close()
            
            return self._process_response(self.last_response)
        except Exception as e:
            return False, {"error": str(e)}


def get_http_tool(base_url: Optional[str] = None, headers: Optional[Dict[str, str]] = None) -> HTTPTool:
    """Factory function to get an HTTP tool instance"""
    return HTTPTool(base_url, headers)"""
