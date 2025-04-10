#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("repl_adapter")

# Initialize the tool registry
TOOL_REGISTRY = {}

def debug_tool_registry(module_name: str = None) -> Dict[str, Any]:
    """
    Debug the tool registry by examining available registered tools
    
    Args:
        module_name: Optional filter for specific module prefix
    
    Returns:
        Dict with debug information
    """
    try:
        # Try to access the GLOBAL_TOOL_REGISTRY from the main module
        import __main__
        if hasattr(__main__, 'GLOBAL_TOOL_REGISTRY'):
            global_registry = __main__.GLOBAL_TOOL_REGISTRY
            
            # Get list of all tools
            all_tools = list(global_registry.keys())
            
            # Filter tools if module_name provided
            if module_name:
                filtered_tools = [t for t in all_tools if t.startswith(f"{module_name}:")]
            else:
                filtered_tools = all_tools
            
            # Test a sample tool if available
            test_result = None
            if "optimization:list_patterns" in global_registry:
                try:
                    test_result = global_registry["optimization:list_patterns"]()
                    logger.info(f"Called optimization:list_patterns: {test_result}")
                except Exception as e:
                    logger.error(f"Error calling optimization:list_patterns: {str(e)}")
                    test_result = {"error": str(e)}
            
            return {
                "status": "success",
                "registry_found": True,
                "total_tools": len(all_tools),
                "filtered_tools": filtered_tools,
                "filtered_count": len(filtered_tools),
                "test_result": test_result
            }
        else:
            logger.warning("GLOBAL_TOOL_REGISTRY not found in __main__")
            return {
                "status": "error",
                "registry_found": False,
                "error": "GLOBAL_TOOL_REGISTRY not found in __main__"
            }
    except Exception as e:
        logger.error(f"Error examining tool registry: {str(e)}")
        return {
            "status": "error",
            "registry_found": False,
            "error": str(e)
        }

def execute_repl(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a REPL environment and return the result
    
    Args:
        code: Python code to execute
    
    Returns:
        Dict with execution result
    """
    try:
        # Create a local scope for execution
        local_scope = {}
        
        # Execute the code
        exec(code, globals(), local_scope)
        
        # Capture any printed output
        return {
            "status": "success",
            "result": "Code executed successfully"
        }
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Register tools
TOOL_REGISTRY["repl:debug_registry"] = debug_tool_registry
TOOL_REGISTRY["repl:execute"] = execute_repl

# Print initialization message
print("✅ REPL diagnostic tools registered successfully")
