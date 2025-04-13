#!/usr/bin/env python3

import os
import json
import re
import datetime
from typing import Dict, Any, List, Optional, Set

def log_api(agent_name, prompt, response):
    """
    Simple function to log API calls and responses to a file.
    Creates a logs directory if it doesn't exist.
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate timestamp and log filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(logs_dir, f"{agent_name}_api.log")
    
    # Write the log entry
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n==== API CALL AT {timestamp} ====\n")
        f.write(f"PROMPT:\n{prompt}")
        f.write(f"\n\n==== API RESPONSE ====\n")
        f.write(f"{response}")
        f.write("\n\n")

def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text, handling various formats"""
    # Try parsing the entire content as JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON within code blocks
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find anything that looks like a JSON object
    object_pattern = r'({[\s\S]*?})'
    match = re.search(object_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # If all extraction attempts fail, return an error object
    return {"error": "Could not extract valid JSON from response", "text": text[:500]}

def extract_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    """Extract all tool calls from a response"""
    tool_usage_pattern = r"I need to use the tool: ([a-zA-Z0-9_:]+)\s*\nParameters:\s*\{([^}]+)\}"
    tool_calls = []
    
    # Find all matches
    matches = re.finditer(tool_usage_pattern, response_content, re.DOTALL)
    for match in matches:
        tool_name = match.group(1).strip()
        params_text = "{" + match.group(2) + "}"
        try:
            params = json.loads(params_text)
            tool_calls.append({
                "tool_name": tool_name,
                "params": params,
                "full_text": match.group(0)
            })
        except json.JSONDecodeError:
            # Skip invalid JSON
            continue
    
    return tool_calls

def process_single_tool_call(response_content):
    """Process a response that may contain a single tool call"""
    tool_usage_pattern = r"I need to use the tool: ([a-zA-Z0-9_:]+)\s*\nParameters:\s*\{([^}]+)\}"
    match = re.search(tool_usage_pattern, response_content, re.DOTALL)
    
    if not match:
        return None, response_content
    
    tool_name = match.group(1).strip()
    params_text = "{" + match.group(2) + "}"
    
    try:
        params = json.loads(params_text)
        # Return the tool call info
        return {
            "tool_name": tool_name,
            "params": params
        }, response_content
    except json.JSONDecodeError:
        return None, response_content

def is_hashable(obj):
    """Check if an object can be used as a dictionary key"""
    try:
        hash(obj)
        return True
    except TypeError:
        return False

def get_config():
    """Get configuration or create default config"""
    import os
    
    try:
        from config import CONFIG
    except ImportError:
        # Create default config if not available
        current_dir = os.path.dirname(os.path.abspath(__file__))
        CONFIG = {
            "api_key": os.environ.get("API_KEY", ""),
            "endpoint": "https://api.anthropic.com/v1/messages",
            "default_model": "claude-3-opus-20240229",
            "output_dir": os.path.join(current_dir, "outputs")
        }
        # Ensure output directory exists
        os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    return CONFIG
