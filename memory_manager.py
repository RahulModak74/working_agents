#!/usr/bin/env python3

import json
import sqlite3
import uuid
import datetime
from typing import List, Dict, Any, Optional

from config import CONFIG


class MemoryManager:
    """Manages persistent memory across agent runs and sessions"""
    
    def __init__(self, db_path: str = CONFIG["memory_db"]):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            agent TEXT,
            timestamp TEXT,
            content TEXT,
            metadata TEXT
        )
        ''')
        
        # Create conversations table for storing conversation context
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            memory_id TEXT,
            timestamp TEXT,
            messages TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_memory(self, agent: str, content: Any, memory_id: str = None, 
                     metadata: Dict[str, Any] = None) -> str:
        """Store a memory entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entry_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        memory_id = memory_id or "default"
        metadata = json.dumps(metadata or {})
        
        # Convert content to JSON string if it's not already a string
        if not isinstance(content, str):
            content = json.dumps(content)
        
        cursor.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)",
            (entry_id, memory_id, agent, timestamp, content, metadata)
        )
        
        conn.commit()
        conn.close()
        
        return entry_id
    
    def get_memories(self, memory_id: str = None, agent: str = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve memories based on filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        query = "SELECT * FROM memories WHERE 1=1"
        params = []
        
        if memory_id:
            query += " AND memory_id = ?"
            params.append(memory_id)
        
        if agent:
            query += " AND agent = ?"
            params.append(agent)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        
        # Parse content and metadata from JSON
        for row in results:
            try:
                row["content"] = json.loads(row["content"])
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
            
            try:
                row["metadata"] = json.loads(row["metadata"])
            except json.JSONDecodeError:
                row["metadata"] = {}
        
        conn.close()
        return results
    
    def store_conversation(self, messages: List[Dict[str, Any]], memory_id: str = None) -> str:
        """Store a conversation context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        memory_id = memory_id or "default"
        
        cursor.execute(
            "INSERT INTO conversations VALUES (?, ?, ?, ?)",
            (conversation_id, memory_id, timestamp, json.dumps(messages))
        )
        
        conn.commit()
        conn.close()
        
        return conversation_id
    
    def get_conversation(self, memory_id: str = None) -> List[Dict[str, Any]]:
        """Retrieve the most recent conversation for a memory ID"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM conversations WHERE memory_id = ? ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query, (memory_id or "default",))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row["messages"])
        return []