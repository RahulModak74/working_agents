#!/usr/bin/env python3

import sys
import os

# Ensure the main directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import your SQL adapter module
from sql_tools_adapter import sql_tables, sql_query, sql_schema

# Test sql:tables
print("\nTesting sql:tables...")
tables_result = sql_tables(database="test_sqlite.db")
print(f"Result: {tables_result}")

# Test sql:schema
print("\nTesting sql:schema...")
schema_result = sql_schema(database="test_sqlite.db", table="users")
print(f"Result: {schema_result}")

# Test sql:query
print("\nTesting sql:query...")
query_result = sql_query(database="test_sqlite.db", query="SELECT * FROM users LIMIT 3")
print(f"Result: {query_result}")
