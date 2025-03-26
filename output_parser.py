#!/usr/bin/env python3

import re
import json
import jsonschema
from typing import Dict, Any, Tuple


class OutputParser:
    """Handles parsing and validation of structured outputs"""
    
    @staticmethod
    def parse_json(text: str) -> Dict[str, Any]:
        """Extract JSON from text, even if it's embedded in other text"""
        # Try to find JSON pattern in the text
        json_pattern = r'```(?:json)?\s*({[\s\S]*?})\s*```|({[\s\S]*?})'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1) or match.group(2)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # If no pattern match, try to parse the entire text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Could not parse JSON from text: {text[:100]}...")
    
    @staticmethod
    def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate JSON data against a schema"""
        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, "Valid"
        except jsonschema.exceptions.ValidationError as e:
            return False, str(e)
    
    @staticmethod
    def extract_format(text: str, output_format: Dict[str, Any]) -> Any:
        """Extract and validate output according to the specified format"""
        format_type = output_format.get("type", "text")
        
        if format_type == "json":
            data = OutputParser.parse_json(text)
            
            # Convert simple schema to jsonschema format
            if "schema" in output_format:
                schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for key, value_type in output_format["schema"].items():
                    if value_type == "string":
                        schema["properties"][key] = {"type": "string"}
                    elif value_type == "number":
                        schema["properties"][key] = {"type": "number"}
                    elif isinstance(value_type, list) and len(value_type) > 0:
                        schema["properties"][key] = {
                            "type": "array",
                            "items": {"type": value_type[0]}
                        }
                    schema["required"].append(key)
                
                valid, message = OutputParser.validate_schema(data, schema)
                if not valid:
                    print(f"Warning: Output does not match schema: {message}")
            
            return data
        
        elif format_type == "markdown":
            # For markdown, just return the text but we could add section extraction
            if "sections" in output_format:
                # Check if all required sections are present
                for section in output_format["sections"]:
                    if not re.search(rf"#{{{1,6}}}\s*{section}", text, re.IGNORECASE):
                        print(f"Warning: Missing section '{section}' in markdown output")
            
            return text
        
        # Default to returning the raw text
        return text