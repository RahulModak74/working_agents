#!/usr/bin/env python3

import clickhouse_driver
import numpy as np
import pandas as pd
import json
import os
import uuid
import hashlib
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from urllib.parse import urlparse, parse_qs

# Import for text encoding
from sentence_transformers import SentenceTransformer

class ClickHouseSQLVectorTool:
    """
    Combined SQL and Vector Database tool using ClickHouse
    Provides SQL querying and vector similarity search in a single tool
    """
    
    def __init__(self, 
                 connection_url: str = None, 
                 model_name: str = "all-MiniLM-L6-v2",
                 vector_dim: int = None,
                 user: str = None,
                 password: str = None,
                 database: str = "default"):
        """
        Initialize with ClickHouse connection parameters
        
        Args:
            connection_url: ClickHouse connection URL (e.g., "clickhouse://host:port/database")
            model_name: Sentence transformer model for text encoding
            vector_dim: Vector dimension (detected automatically if not provided)
            user: ClickHouse username
            password: ClickHouse password
            database: ClickHouse database name
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.encoder = None
        self.client = None
        self.history = []
        
        # Parse connection URL if provided
        if connection_url and connection_url.startswith("clickhouse://"):
            parsed_url = urlparse(connection_url)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 9000
            user = parsed_url.username or user or "default"
            password = parsed_url.password or password or ""
            database = parsed_url.path.lstrip('/') or database
            
            # Parse query parameters
            if parsed_url.query:
                query_params = parse_qs(parsed_url.query)
                if 'secure' in query_params:
                    secure = query_params['secure'][0].lower() in ('true', '1', 'yes')
                else:
                    secure = False
            else:
                secure = False
        else:
            # Default connection parameters
            host = "localhost"
            port = 9000
            user = user or "default"
            password = password or ""
            secure = False
            
        # Initialize connection
        self.connect(host=host, port=port, user=user, password=password, 
                    database=database, secure=secure)
    
    def connect(self, host: str = "localhost", port: int = 9000, 
               user: str = "default", password: str = "", 
               database: str = "default", secure: bool = False) -> bool:
        """
        Connect to ClickHouse server
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = clickhouse_driver.Client(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                secure=secure
            )
            # Test connection
            self.client.execute("SELECT 1")
            return True
        except Exception as e:
            print(f"Error connecting to ClickHouse: {e}")
            self.client = None
            return False
    
    def disconnect(self):
        """Close the ClickHouse connection"""
        self.client = None
    
    def _load_encoder(self):
        """Load the sentence transformer model for text encoding"""
        if self.encoder is None:
            try:
                self.encoder = SentenceTransformer(self.model_name)
                # Get vector dimension from the model if not specified
                if self.vector_dim is None:
                    self.vector_dim = self.encoder.get_sentence_embedding_dimension()
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Encode text into vector embedding
        
        Args:
            text: Text to encode
            
        Returns:
            List[float]: Vector embedding
        """
        self._load_encoder()
        return self.encoder.encode(text).tolist()
    
    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encode multiple texts into vector embeddings
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List[List[float]]: List of vector embeddings
        """
        self._load_encoder()
        return self.encoder.encode(texts).tolist()
    
    def ensure_vector_table(self, table_name: str) -> bool:
        """
        Ensure vector table exists with proper structure
        
        Args:
            table_name: Name of the vector table
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            return False
            
        try:
            # Load encoder to ensure vector_dim is defined
            self._load_encoder()
            
            # Create table if not exists
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id String,
                text String,
                metadata String,
                embedding Array(Float32),
                created_at DateTime DEFAULT now()
            )
            ENGINE = MergeTree()
            ORDER BY id
            """
            
            self.client.execute(create_query)
            
            # Create cosine distance function if it doesn't exist
            # This may fail if already exists, which is fine
            try:
                self.client.execute("""
                CREATE FUNCTION IF NOT EXISTS cosineDistance AS (a, b) -> 
                    1 - sumProduct(a, b) / (sqrt(sumProduct(a, a)) * sqrt(sumProduct(b, b)));
                """)
            except:
                pass
                
            return True
        except Exception as e:
            print(f"Error creating vector table: {e}")
            return False
    
    def execute_query(self, query: str, params: Dict[str, Any] = None) -> Tuple[bool, Union[List[Dict[str, Any]], str]]:
        """
        Execute a SQL query in ClickHouse
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Tuple[bool, Union[List[Dict[str, Any]], str]]: 
                (Success flag, Results as list of dicts or info message)
        """
        if not self.client:
            return False, "Not connected to ClickHouse"
        
        try:
            # Use with_column_types to get column names
            result, columns = self.client.execute(
                query, 
                params or {}, 
                with_column_types=True
            )
            
            # For SELECT queries, convert to list of dicts
            if query.strip().upper().startswith("SELECT"):
                column_names = [col[0] for col in columns]
                results = []
                for row in result:
                    # Convert row values to JSON-compatible types
                    processed_row = {}
                    for i, value in enumerate(row):
                        # Handle numpy arrays and other special types
                        if isinstance(value, np.ndarray):
                            processed_row[column_names[i]] = value.tolist()
                        elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                            processed_row[column_names[i]] = int(value)
                        elif isinstance(value, (np.float64, np.float32, np.float16)):
                            processed_row[column_names[i]] = float(value)
                        else:
                            processed_row[column_names[i]] = value
                    results.append(processed_row)
                return True, results
            else:
                # For non-SELECT queries, return row count
                count = len(result) if result else 0
                return True, f"Query executed successfully. Rows affected: {count}"
        except Exception as e:
            return False, f"Error executing query: {str(e)}"
    
    def get_tables(self) -> List[str]:
        """
        Get a list of tables in the current database
        
        Returns:
            List[str]: List of table names
        """
        if not self.client:
            return []
        
        try:
            result, _ = self.client.execute(
                "SHOW TABLES", 
                with_column_types=True
            )
            return [row[0] for row in result]
        except Exception as e:
            print(f"Error getting tables: {e}")
            return []
    
    def get_schema(self, table_name: str = None) -> Dict[str, Any]:
        """
        Get the schema for the specified table or all tables
        
        Args:
            table_name: Name of the table, or None for all tables
            
        Returns:
            Dict[str, Any]: Table schema information
        """
        if not self.client:
            return {}
        
        schema_info = {}
        
        try:
            if table_name:
                tables = [table_name]
            else:
                tables = self.get_tables()
            
            for table in tables:
                try:
                    # Get column information
                    result, _ = self.client.execute(
                        f"DESCRIBE TABLE {table}",
                        with_column_types=True
                    )
                    
                    columns = []
                    for row in result:
                        columns.append({
                            "name": row[0],
                            "type": row[1],
                            "default_type": row[2],
                            "default_expression": row[3]
                        })
                    
                    schema_info[table] = columns
                except:
                    # Skip tables that can't be described
                    continue
            
            return schema_info
        except Exception as e:
            print(f"Error getting schema: {e}")
            return {}
    
    def add_vector(self, table_name: str, text: str, metadata: Dict[str, Any] = None, 
                  doc_id: str = None, create_table: bool = True) -> str:
        """
        Add a vector to the database
        
        Args:
            table_name: Table to add the vector to
            text: Text to encode and store
            metadata: Additional metadata to store with the vector
            doc_id: Optional document ID (generated if not provided)
            create_table: Create the table if it doesn't exist
            
        Returns:
            str: Document ID of the added vector
        """
        if not self.client:
            return None
            
        # Create table if needed
        if create_table:
            self.ensure_vector_table(table_name)
            
        # Generate ID if not provided
        doc_id = doc_id or str(uuid.uuid4())
        
        # Encode text to vector
        embedding = self._encode_text(text)
        
        # Convert metadata to JSON string
        metadata_str = json.dumps(metadata or {})
        
        try:
            # Insert the vector
            self.client.execute(
                f"""
                INSERT INTO {table_name} 
                (id, text, metadata, embedding)
                VALUES
                """,
                [(doc_id, text, metadata_str, embedding)]
            )
            return doc_id
        except Exception as e:
            print(f"Error adding vector: {e}")
            return None
    
    def add_vectors(self, table_name: str, texts: List[str], 
                   metadatas: List[Dict[str, Any]] = None,
                   doc_ids: List[str] = None, create_table: bool = True) -> List[str]:
        """
        Add multiple vectors to the database
        
        Args:
            table_name: Table to add the vectors to
            texts: List of texts to encode and store
            metadatas: List of metadata dicts to store
            doc_ids: Optional list of document IDs
            create_table: Create the table if it doesn't exist
            
        Returns:
            List[str]: Document IDs of the added vectors
        """
        if not self.client:
            return []
            
        # Create table if needed
        if create_table:
            self.ensure_vector_table(table_name)
            
        # Generate IDs if not provided
        if not doc_ids:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
        elif len(doc_ids) < len(texts):
            # Fill in missing IDs
            doc_ids.extend([str(uuid.uuid4()) for _ in range(len(texts) - len(doc_ids))])
            
        # Ensure metadatas is properly sized
        if not metadatas:
            metadatas = [{} for _ in texts]
        elif len(metadatas) < len(texts):
            metadatas.extend([{} for _ in range(len(texts) - len(metadatas))])
            
        # Encode texts to vectors
        embeddings = self._encode_batch(texts)
        
        # Convert metadatas to JSON strings
        metadata_strs = [json.dumps(m) for m in metadatas]
        
        try:
            # Insert the vectors
            data = list(zip(doc_ids, texts, metadata_strs, embeddings))
            self.client.execute(
                f"""
                INSERT INTO {table_name} 
                (id, text, metadata, embedding)
                VALUES
                """,
                data
            )
            return doc_ids
        except Exception as e:
            print(f"Error adding vectors: {e}")
            return []
    
    def search_vectors(self, table_name: str, query_text: str, limit: int = 5, 
                      filter_condition: str = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using cosine similarity
        
        Args:
            table_name: Table to search in
            query_text: Text to search for
            limit: Maximum number of results to return
            filter_condition: Optional SQL WHERE condition for filtering
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        if not self.client:
            return []
            
        # Encode query text
        query_embedding = self._encode_text(query_text)
        
        # Build the query
        query = f"""
        SELECT 
            id, 
            text, 
            metadata,
            cosineDistance(embedding, {query_embedding}) AS distance,
            1 / (1 + cosineDistance(embedding, {query_embedding})) AS similarity
        FROM {table_name}
        """
        
        if filter_condition:
            query += f" WHERE {filter_condition}"
            
        query += " ORDER BY distance ASC LIMIT " + str(limit)
        
        # Execute the query
        success, results = self.execute_query(query)
        
        if not success:
            return []
            
        # Process results
        processed_results = []
        for row in results:
            try:
                # Parse metadata JSON
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                
                processed_results.append({
                    "id": row["id"],
                    "text": row["text"],
                    "metadata": metadata,
                    "distance": float(row["distance"]),
                    "similarity": float(row["similarity"])
                })
            except Exception as e:
                print(f"Error processing result row: {e}")
                
        return processed_results
    
    def delete_vector(self, table_name: str, doc_id: str) -> bool:
        """
        Delete a vector from the database
        
        Args:
            table_name: Table to delete from
            doc_id: Document ID to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            return False
            
        try:
            self.client.execute(
                f"ALTER TABLE {table_name} DELETE WHERE id = %s",
                [doc_id]
            )
            return True
        except Exception as e:
            print(f"Error deleting vector: {e}")
            return False
    
    def import_from_csv(self, file_path: str, table_name: str, if_exists: str = 'append',
                       text_column: str = None, metadata_columns: List[str] = None,
                       id_column: str = None, create_embeddings: bool = True) -> Tuple[bool, str]:
        """
        Import data from a CSV file into a vector table
        
        Args:
            file_path: Path to CSV file
            table_name: Table to import into
            if_exists: What to do if table exists ('append' or 'replace')
            text_column: Column to use for text embedding
            metadata_columns: Columns to include in metadata
            id_column: Column to use as document ID
            create_embeddings: Whether to create embeddings for the text
            
        Returns:
            Tuple[bool, str]: (Success flag, Information message)
        """
        if not self.client:
            return False, "Not connected to ClickHouse"
            
        if not os.path.exists(file_path):
            return False, f"File not found: {file_path}"
            
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            if if_exists == 'replace':
                # Drop and recreate table
                self.client.execute(f"DROP TABLE IF EXISTS {table_name}")
                self.ensure_vector_table(table_name)
            else:
                # Make sure table exists
                self.ensure_vector_table(table_name)
                
            # Process data
            doc_ids = []
            texts = []
            metadatas = []
            
            for _, row in df.iterrows():
                # Get document ID
                if id_column and id_column in row:
                    doc_id = str(row[id_column])
                else:
                    doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                # Get text for embedding
                if text_column and text_column in row:
                    text = str(row[text_column])
                else:
                    # Use the first string column as text
                    text_cols = [col for col in df.columns if df[col].dtype == 'object']
                    if text_cols:
                        text = str(row[text_cols[0]])
                    else:
                        text = ""
                texts.append(text)
                
                # Get metadata
                metadata = {}
                if metadata_columns:
                    for col in metadata_columns:
                        if col in row:
                            metadata[col] = row[col]
                else:
                    # Use all columns as metadata
                    for col in df.columns:
                        metadata[col] = row[col]
                metadatas.append(metadata)
            
            if create_embeddings:
                # Add vectors to database
                result = self.add_vectors(table_name, texts, metadatas, doc_ids)
                return len(result) > 0, f"Imported {len(result)} vectors from CSV"
            else:
                # Insert data without embeddings
                data = []
                for doc_id, text, metadata in zip(doc_ids, texts, metadatas):
                    data.append((doc_id, text, json.dumps(metadata), [0.0] * self.vector_dim))
                
                self.client.execute(
                    f"""
                    INSERT INTO {table_name} 
                    (id, text, metadata, embedding)
                    VALUES
                    """,
                    data
                )
                return True, f"Imported {len(data)} rows from CSV (without embeddings)"
                
        except Exception as e:
            return False, f"Error importing from CSV: {str(e)}"
    
    def hybrid_search(self, table_name: str, query_text: str, sql_condition: str = None, 
                     limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search using both SQL filtering and vector similarity
        
        Args:
            table_name: Table to search in
            query_text: Text to search for
            sql_condition: SQL WHERE condition for filtering
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        return self.search_vectors(table_name, query_text, limit, sql_condition)
    
    def export_to_csv(self, query: str, output_file: str) -> Tuple[bool, str]:
        """
        Export query results to a CSV file
        
        Args:
            query: SQL query to execute
            output_file: Path to output CSV file
            
        Returns:
            Tuple[bool, str]: (Success flag, Information message)
        """
        success, results = self.execute_query(query)
        
        if not success:
            return False, f"Query failed: {results}"
            
        try:
            if not isinstance(results, list):
                return False, "Query did not return a result set"
                
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            return True, f"Exported {len(results)} rows to {output_file}"
        except Exception as e:
            return False, f"Error exporting to CSV: {str(e)}"
    
    def optimize_table(self, table_name: str) -> bool:
        """
        Optimize a table for better performance
        
        Args:
            table_name: Table to optimize
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            return False
            
        try:
            self.client.execute(f"OPTIMIZE TABLE {table_name} FINAL")
            return True
        except Exception as e:
            print(f"Error optimizing table: {e}")
            return False
            
    def create_table_with_vectors(self, table_name: str, columns: List[Dict[str, str]], 
                                primary_key: str = None) -> bool:
        """
        Create a custom table that includes vector embedding capability
        
        Args:
            table_name: Name of the table to create
            columns: List of column definitions (name, type)
            primary_key: Primary key column name
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            return False
            
        try:
            # Load encoder to ensure vector_dim is defined
            self._load_encoder()
            
            # Build column definitions
            column_defs = []
            for col in columns:
                column_defs.append(f"{col['name']} {col['type']}")
                
            # Always add embedding column
            column_defs.append(f"embedding Array(Float32)")
            
            # Set primary key
            pk = primary_key or "id"
            
            # Create table
            create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(column_defs)}
            )
            ENGINE = MergeTree()
            ORDER BY {pk}
            """
            
            self.client.execute(create_query)
            return True
        except Exception as e:
            print(f"Error creating table: {e}")
            return False


# Factory function to create ClickHouseSQLVectorTool instance
def get_clickhouse_sql_vector_tool(
        connection_url: str = None,
        model_name: str = "all-MiniLM-L6-v2",
        vector_dim: int = None,
        user: str = None,
        password: str = None,
        database: str = "default") -> ClickHouseSQLVectorTool:
    return ClickHouseSQLVectorTool(
        connection_url=connection_url,
        model_name=model_name,
        vector_dim=vector_dim,
        user=user,
        password=password,
        database=database
    )

# === ADD THIS NEW SECTION BELOW THE FACTORY FUNCTION ===

# Handler functions for TOOL_REGISTRY
def clickhouse_query_handler(query, params=None):
    tool = get_clickhouse_sql_vector_tool()
    return tool.execute_query(query, params)

def clickhouse_vector_search_handler(table_name, query_text, limit=5):
    tool = get_clickhouse_sql_vector_tool()
    return tool.search_vectors_
# Explicit TOOL_REGISTRY definition
TOOL_REGISTRY = {
    "clickhouse:query": clickhouse_query_handler,
    "clickhouse:vector_search": clickhouse_vector_search_handler,
}
