#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("memory_adapter")

# Initialize the tool registry (required by universal_main.py)
TOOL_REGISTRY = {}

# Configuration
class MemoryConfig:
    def __init__(self):
        self.storage_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_storage")
        self.default_memory_id = "default_memory"
        
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)

# Global configuration
MEMORY_CONFIG = MemoryConfig()

class MemoryManager:
    def __init__(self, config=None):
        self.config = config or MEMORY_CONFIG
        self.active_memories = {}
        
    def create_system(self, system_id: str = None, description: str = None) -> Dict[str, Any]:
        """Create a new memory system"""
        if not system_id:
            system_id = self.config.default_memory_id
        
        # Check if memory system already exists
        if system_id in self.active_memories:
            return {
                "status": "warning",
                "message": f"Memory system '{system_id}' already exists and will be reused.",
                "memory_system_id": system_id
            }
        
        # Create new memory system
        memory_file = os.path.join(self.config.storage_dir, f"{system_id}.json")
        
        # Initialize memory structure
        memory_system = {
            "id": system_id,
            "description": description or f"Memory system created on {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            "entities": {},
            "concepts": {},
            "relations": [],
            "operations": []
        }
        
        # Save to file
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_system, f, indent=2)
            
            # Add to active memories
            self.active_memories[system_id] = memory_system
            
            return {
                "status": "success",
                "message": f"Memory system '{system_id}' created successfully.",
                "memory_system_id": system_id
            }
        except Exception as e:
            logger.error(f"Error creating memory system: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to create memory system: {str(e)}"
            }
    
    def store_operation(self, memory_id: str = None, operation_type: str = "store", 
                        content: Any = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store information in the memory system"""
        if not memory_id:
            memory_id = self.config.default_memory_id
        
        # Create memory system if it doesn't exist
        if memory_id not in self.active_memories:
            self.create_system(memory_id)
        
        # Retrieve memory system
        memory_system = self.active_memories.get(memory_id)
        if not memory_system:
            memory_file = os.path.join(self.config.storage_dir, f"{memory_id}.json")
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_system = json.load(f)
                    self.active_memories[memory_id] = memory_system
                except Exception as e:
                    logger.error(f"Error loading memory system: {str(e)}")
                    return {
                        "status": "error",
                        "error": f"Failed to load memory system: {str(e)}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Memory system '{memory_id}' does not exist."
                }
        
        # Create operation
        operation = {
            "id": f"op_{len(memory_system['operations']) + 1}",
            "type": operation_type,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "content": content,
            "metadata": metadata or {}
        }
        
        # Add operation to memory system
        memory_system['operations'].append(operation)
        memory_system['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update memory file
        memory_file = os.path.join(self.config.storage_dir, f"{memory_id}.json")
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_system, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Operation stored successfully in memory system '{memory_id}'.",
                "operation_id": operation["id"]
            }
        except Exception as e:
            logger.error(f"Error updating memory system: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to update memory system: {str(e)}"
            }
    
    def retrieve_operation(self, memory_id: str = None, operation_id: str = None) -> Dict[str, Any]:
        """Retrieve a specific operation from the memory system"""
        if not memory_id:
            memory_id = self.config.default_memory_id
        
        # Retrieve memory system
        memory_system = self.active_memories.get(memory_id)
        if not memory_system:
            memory_file = os.path.join(self.config.storage_dir, f"{memory_id}.json")
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_system = json.load(f)
                    self.active_memories[memory_id] = memory_system
                except Exception as e:
                    logger.error(f"Error loading memory system: {str(e)}")
                    return {
                        "status": "error",
                        "error": f"Failed to load memory system: {str(e)}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Memory system '{memory_id}' does not exist."
                }
        
        # Retrieve operation
        if operation_id:
            for operation in memory_system['operations']:
                if operation['id'] == operation_id:
                    return {
                        "status": "success",
                        "operation": operation
                    }
            
            return {
                "status": "error",
                "error": f"Operation '{operation_id}' not found in memory system '{memory_id}'."
            }
        else:
            # Return the most recent operation
            if memory_system['operations']:
                return {
                    "status": "success",
                    "operation": memory_system['operations'][-1]
                }
            else:
                return {
                    "status": "error",
                    "error": f"No operations found in memory system '{memory_id}'."
                }
    
    def retrieve_all(self, memory_id: str = None, operation_type: str = None, 
                    limit: int = None) -> Dict[str, Any]:
        """Retrieve all operations from the memory system"""
        if not memory_id:
            memory_id = self.config.default_memory_id
        
        # Retrieve memory system
        memory_system = self.active_memories.get(memory_id)
        if not memory_system:
            memory_file = os.path.join(self.config.storage_dir, f"{memory_id}.json")
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_system = json.load(f)
                    self.active_memories[memory_id] = memory_system
                except Exception as e:
                    logger.error(f"Error loading memory system: {str(e)}")
                    return {
                        "status": "error",
                        "error": f"Failed to load memory system: {str(e)}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Memory system '{memory_id}' does not exist."
                }
        
        # Filter operations by type if specified
        operations = memory_system['operations']
        if operation_type:
            operations = [op for op in operations if op['type'] == operation_type]
        
        # Limit the number of operations if specified
        if limit and limit > 0:
            operations = operations[-limit:]
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "operations_count": len(operations),
            "operations": operations
        }
    
    def submit_result(self, memory_id: str = None, result: Any = None, 
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Submit a result to the memory system"""
        return self.store_operation(memory_id, "result", result, metadata)
    
    def list_patterns(self, pattern_type: str = None) -> Dict[str, Any]:
        """List available memory patterns"""
        # Define available patterns
        patterns = {
            "working_memory": {
                "description": "Short-term memory for active processing and reasoning",
                "use_cases": ["Temporary storage during reasoning", "Active manipulation of data"]
            },
            "episodic_memory": {
                "description": "Storage for specific events and experiences",
                "use_cases": ["Remembering past research findings", "Learning from historical interactions"]
            },
            "semantic_memory": {
                "description": "Storage for facts, concepts, and general knowledge",
                "use_cases": ["Domain knowledge representation", "Concept hierarchies"]
            },
            "procedural_memory": {
                "description": "Storage for skills, procedures, and actions",
                "use_cases": ["Research methodologies", "Workflow patterns"]
            },
            "associative_memory": {
                "description": "Connects related items through associations",
                "use_cases": ["Finding related concepts", "Pattern recognition"]
            },
            "distributed_memory": {
                "description": "Stores information across multiple locations",
                "use_cases": ["Redundant storage", "Parallel processing"]
            }
        }
        
        # Filter by pattern type if specified
        if pattern_type and pattern_type in patterns:
            return {
                "status": "success",
                "pattern": pattern_type,
                "details": patterns[pattern_type]
            }
        
        return {
            "status": "success",
            "available_patterns": list(patterns.keys()),
            "patterns": patterns
        }
    
    def get_memory_stats(self, memory_id: str = None) -> Dict[str, Any]:
        """Get statistics about a memory system"""
        if not memory_id:
            memory_id = self.config.default_memory_id
        
        # Retrieve memory system
        memory_system = self.active_memories.get(memory_id)
        if not memory_system:
            memory_file = os.path.join(self.config.storage_dir, f"{memory_id}.json")
            if os.path.exists(memory_file):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        memory_system = json.load(f)
                    self.active_memories[memory_id] = memory_system
                except Exception as e:
                    logger.error(f"Error loading memory system: {str(e)}")
                    return {
                        "status": "error",
                        "error": f"Failed to load memory system: {str(e)}"
                    }
            else:
                return {
                    "status": "error",
                    "error": f"Memory system '{memory_id}' does not exist."
                }
        
        # Calculate statistics
        operation_counts = {}
        for operation in memory_system['operations']:
            operation_type = operation['type']
            operation_counts[operation_type] = operation_counts.get(operation_type, 0) + 1
        
        return {
            "status": "success",
            "memory_id": memory_id,
            "description": memory_system['description'],
            "created_at": memory_system['created_at'],
            "last_updated": memory_system['last_updated'],
            "total_operations": len(memory_system['operations']),
            "operation_types": operation_counts,
            "entities_count": len(memory_system['entities']),
            "concepts_count": len(memory_system['concepts']),
            "relations_count": len(memory_system['relations'])
        }

# Global memory manager
MEMORY_MANAGER = MemoryManager()

#---------------------------
# Tool Registration Functions
#---------------------------

def memory_create_system(system_id: str = None, description: str = None) -> Dict[str, Any]:
    """
    Create a new memory system.
    
    Args:
        system_id: Optional identifier for the memory system
        description: Optional description of the memory system
    
    Returns:
        Dict with creation status and memory system ID
    """
    return MEMORY_MANAGER.create_system(system_id, description)

def memory_store_operation(memory_id: str = None, operation_type: str = "store", 
                          content: Any = None, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Store information in the memory system.
    
    Args:
        memory_id: Memory system identifier
        operation_type: Type of operation to store
        content: Content to store
        metadata: Additional metadata
    
    Returns:
        Dict with storage status and operation ID
    """
    return MEMORY_MANAGER.store_operation(memory_id, operation_type, content, metadata)

def memory_retrieve_operation(memory_id: str = None, operation_id: str = None) -> Dict[str, Any]:
    """
    Retrieve a specific operation from the memory system.
    
    Args:
        memory_id: Memory system identifier
        operation_id: Operation identifier to retrieve
    
    Returns:
        Dict with retrieval status and operation details
    """
    return MEMORY_MANAGER.retrieve_operation(memory_id, operation_id)

def memory_retrieve_all(memory_id: str = None, operation_type: str = None, 
                       limit: int = None) -> Dict[str, Any]:
    """
    Retrieve all operations from the memory system.
    
    Args:
        memory_id: Memory system identifier
        operation_type: Optional type of operations to retrieve
        limit: Optional limit on the number of operations to retrieve
    
    Returns:
        Dict with retrieval status and operations
    """
    return MEMORY_MANAGER.retrieve_all(memory_id, operation_type, limit)

def memory_submit_result(memory_id: str = None, result: Any = None, 
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit a result to the memory system.
    
    Args:
        memory_id: Memory system identifier
        result: Result to submit
        metadata: Additional metadata
    
    Returns:
        Dict with submission status and operation ID
    """
    return MEMORY_MANAGER.submit_result(memory_id, result, metadata)

def memory_list_patterns(pattern_type: str = None) -> Dict[str, Any]:
    """
    List available memory patterns.
    
    Args:
        pattern_type: Optional specific pattern to retrieve details for
    
    Returns:
        Dict with available patterns or specific pattern details
    """
    return MEMORY_MANAGER.list_patterns(pattern_type)

def memory_get_stats(memory_id: str = None) -> Dict[str, Any]:
    """
    Get statistics about a memory system.
    
    Args:
        memory_id: Memory system identifier
    
    Returns:
        Dict with memory system statistics
    """
    return MEMORY_MANAGER.get_memory_stats(memory_id)

# Register tools
TOOL_REGISTRY["memory:create_system"] = memory_create_system
TOOL_REGISTRY["memory:store_operation"] = memory_store_operation
TOOL_REGISTRY["memory:retrieve_operation"] = memory_retrieve_operation
TOOL_REGISTRY["memory:retrieve_all"] = memory_retrieve_all
TOOL_REGISTRY["memory:submit_result"] = memory_submit_result
TOOL_REGISTRY["memory:list_patterns"] = memory_list_patterns
TOOL_REGISTRY["memory:get_stats"] = memory_get_stats

# Function for executing memory tools
def execute_tool(tool_id: str, **kwargs):
    """Execute a memory tool based on the tool ID"""
    if tool_id in TOOL_REGISTRY:
        handler = TOOL_REGISTRY[tool_id]
        return handler(**kwargs)
    else:
        return {"error": f"Unknown tool: {tool_id}"}

# Print initialization message
print("✅ Memory management tools registered successfully")
