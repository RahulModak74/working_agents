#!/usr/bin/env python3

import sys
import os
import json
import subprocess
import importlib.util
from typing import Dict, Any, Optional, Tuple

def run_python_script(script_path: str, input_data: Optional[str] = None, 
                     args: Optional[list] = None) -> Tuple[Dict[str, Any], str]:
    """Execute a Python script and return its output"""
    if not os.path.exists(script_path):
        return {"error": f"Script file not found: {script_path}"}, f"Script file not found: {script_path}"
    
    # Build command
    cmd = [sys.executable, script_path]
    
    # Add input data file if provided
    if input_data and os.path.exists(input_data):
        cmd.append(input_data)
    
    # Add any additional arguments
    if args:
        cmd.extend(args)
    
    try:
        # Execute the script and capture output
        print(f"Executing script: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Check for errors
        if process.returncode != 0:
            error_msg = f"Script execution failed with return code {process.returncode}: {stderr}"
            print(error_msg)
            return {"error": error_msg, "stderr": stderr}, stderr
        
        # Try to parse output as JSON
        output_data = stdout.strip()
        try:
            json_data = json.loads(output_data)
            return json_data, output_data
        except json.JSONDecodeError:
            # Return raw output if not valid JSON
            return {"output": output_data}, output_data
    
    except Exception as e:
        error_msg = f"Error executing script: {str(e)}"
        print(error_msg)
        return {"error": error_msg}, str(e)

def import_and_run_function(script_path: str, function_name: str, 
                          input_data: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Import a Python script and run a specific function from it"""
    if not os.path.exists(script_path):
        return {"error": f"Script file not found: {script_path}"}, f"Script file not found: {script_path}"
    
    try:
        # Import the module
        module_name = os.path.basename(script_path)[:-3]  # Remove .py
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Check if the function exists
        if not hasattr(module, function_name):
            return {"error": f"Function '{function_name}' not found in {script_path}"}, \
                   f"Function '{function_name}' not found in {script_path}"
        
        # Get the function
        func = getattr(module, function_name)
        
        # Call the function with or without input data
        if input_data and os.path.exists(input_data):
            result = func(input_data)
        else:
            result = func()
        
        # Handle different return types
        if isinstance(result, dict):
            return result, json.dumps(result)
        elif isinstance(result, str):
            # Try to parse as JSON if it looks like JSON
            if result.strip().startswith('{') and result.strip().endswith('}'):
                try:
                    json_data = json.loads(result)
                    return json_data, result
                except json.JSONDecodeError:
                    pass
            return {"output": result}, result
        else:
            # Convert other types to string
            str_result = str(result)
            return {"output": str_result}, str_result
            
    except Exception as e:
        error_msg = f"Error importing or executing function: {str(e)}"
        print(error_msg)
        return {"error": error_msg}, str(e)
