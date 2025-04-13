#!/usr/bin/env python3

import os
import sys
import json
import inspect
import importlib.util
import glob
from typing import Dict, Any, List, Optional, Callable, Set
from functools import wraps

class ToolManager:
    """Manages tool discovery, registration, and execution"""
    
    def __init__(self):
        self.tools = {}  # Tool registry (internal use only, not exposed)
        self.imported_modules = set()  # Track imported modules
        self.namespace_prefixes = {}  # Map module to namespace prefix
    
    def discover_tools(self, directories: List[str] = None) -> int:
        """Automatically discover all Python modules and their functions in given directories"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if directories is None:
            directories = [current_dir]
            
            # Also include any directories within current_dir
            for item in os.listdir(current_dir):
                item_path = os.path.join(current_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    directories.append(item_path)
        
        total_tools = 0
        # Find all Python files
        for directory in directories:
            python_files = glob.glob(os.path.join(directory, "*.py"))
            for py_file in python_files:
                if py_file.endswith("auto_runner.py"):
                    continue  # Skip this file
                
                module_name = os.path.basename(py_file)[:-3]  # Remove .py
                
                # Skip if already imported
                if module_name in self.imported_modules:
                    continue
                
                try:
                    # Import the module
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.imported_modules.add(module_name)
                    
                    # Determine namespace prefix
                    if hasattr(module, 'TOOL_NAMESPACE'):
                        prefix = getattr(module, 'TOOL_NAMESPACE')
                    else:
                        # By default, use the module name
                        prefix = module_name
                        
                        # Special case: if it ends with _adapter, remove that part
                        if prefix.endswith("_adapter"):
                            prefix = prefix[:-8]  # Remove "_adapter"
                    
                    self.namespace_prefixes[module_name] = prefix
                    
                    # If the module has TOOL_REGISTRY, import those specific tools first
                    if hasattr(module, 'TOOL_REGISTRY'):
                        for tool_id, tool_handler in getattr(module, 'TOOL_REGISTRY').items():
                            self.register_tool(tool_id, tool_handler)
                            total_tools += 1
                    
                    # Then scan for functions that should be registered as tools
                    module_tools = self._discover_module_tools(module, prefix)
                    total_tools += len(module_tools)
                    
                    print(f"Imported module {module_name} with {len(module_tools)} auto-discovered functions")
                    
                except Exception as e:
                    print(f"Error importing {module_name}: {e}")
        
        return total_tools
    
    def _discover_module_tools(self, module, prefix: str) -> List[str]:
        """Discover and register tools from a module"""
        discovered_tools = []
        
        # Get all functions from the module
        for name, obj in inspect.getmembers(module):
            # Skip private functions, special methods, and non-functions
            if name.startswith('_') or not inspect.isfunction(obj):
                continue
                
            # Skip functions that are already in TOOL_REGISTRY
            if hasattr(module, 'TOOL_REGISTRY') and name in getattr(module, 'TOOL_REGISTRY').values():
                continue
            
            # Check if function has a docstring (we only want documented functions)
            if obj.__doc__:
                # Create a tool ID based on the prefix and function name
                tool_id = f"{prefix}:{name}"
                self.register_tool(tool_id, obj)
                discovered_tools.append(tool_id)
        print (discovered_tools) 
        return discovered_tools
    
    def register_tool(self, tool_id: str, handler: Callable) -> None:
        """Register a function as a tool"""
        @wraps(handler)
        def flexible_handler(**kwargs):
            try:
                # Try to call with all kwargs
                return handler(**kwargs)
            except TypeError:
                # If that fails, filter kwargs to match function signature
                sig = inspect.signature(handler)
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() 
                    if k in sig.parameters
                }
                try:
                    return handler(**filtered_kwargs)
                except Exception as e:
                    return {"error": f"Tool execution failed: {str(e)}"}
        
        self.tools[tool_id] = flexible_handler
    
    def execute_tool(self, tool_id: str, **kwargs) -> Any:
        """Execute a registered tool"""
        if tool_id not in self.tools:
            # Try to find it by prefix
            prefix = tool_id.split(':', 1)[0] if ':' in tool_id else tool_id
            
            # Look for modules that might contain this tool
            for module_name, module_prefix in self.namespace_prefixes.items():
                if module_prefix == prefix and module_name not in self.imported_modules:
                    # Try to import module
                    try:
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        module_path = os.path.join(current_dir, f"{module_name}.py")
                        if not os.path.exists(module_path):
                            continue
                            
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        self._discover_module_tools(module, prefix)
                        self.imported_modules.add(module_name)
                    except Exception:
                        pass
        
        if tool_id not in self.tools:
            return {"error": f"Unknown tool: {tool_id}"}
        
        try:
            print(f"Executing tool {tool_id} with params: {json.dumps(kwargs, indent=2)}")
            result = self.tools[tool_id](**kwargs)
            return result
        except Exception as e:
            print(f"Error executing tool {tool_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_all_tools(self) -> List[str]:
        """Get a list of all registered tool IDs"""
        return list(self.tools.keys())
    
    def is_tool_available(self, tool_id: str) -> bool:
        """Check if a tool is available"""
        return tool_id in self.tools

# Create a global tool manager instance
tool_manager = ToolManager()
