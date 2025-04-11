#!/usr/bin/env python3

import os
import json
import subprocess
import sys
import time

def test_openrouter(api_key=None, model=None, verbose=False):
    """
    Test the OpenRouter API connection and response format
    """
    # Get configuration
    if not api_key:
        # Try to import from config
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from config import CONFIG
            api_key = CONFIG.get("api_key")
        except ImportError:
            # Try environment variable
            api_key = os.environ.get("API_KEY")
    
    if not api_key:
        print("ERROR: No API key provided. Set API_KEY environment variable or use --api-key")
        return False
    
    if not model:
        # Try to import from config
        try:
            from config import CONFIG
            model = CONFIG.get("default_model")
        except ImportError:
            # Default model
            model = "anthropic/claude-3-opus-20240229"
    
    print(f"Testing OpenRouter with model: {model}")
    
    # Create a simple test message
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON."},
        {"role": "user", "content": "Return a simple JSON object with the keys 'status' and 'message'. Set status to 'success' and message to 'API test successful'."}
    ]
    
    # Create the payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 100
    }
    
    # Safely encode the payload to JSON
    payload_str = json.dumps(payload).replace("'", "'\\''")
    
    # Create the curl command
    curl_command = f"""curl https://openrouter.ai/api/v1/chat/completions \\
      -H "Authorization: Bearer {api_key}" \\
      -H "Content-Type: application/json" \\
      -H "HTTP-Referer: http://localhost" \\
      -d '{payload_str}'"""
    
    # Run the command
    if verbose:
        print("\nSending request to OpenRouter...")
        print(f"Payload: {json.dumps(payload, indent=2)}")
    
    output_file = f"openrouter_test_{int(time.time())}.json"
    
    try:
        subprocess.run(curl_command + f" -o {output_file}", shell=True, check=True)
        
        # Read the response
        with open(output_file, 'r', encoding='utf-8') as f:
            response = f.read()
        
        # Parse the response
        try:
            json_response = json.loads(response)
            
            print("\n--- API Response Structure ---")
            if verbose:
                print(json.dumps(json_response, indent=2))
            else:
                # Print the top-level keys
                print(f"Top-level keys: {list(json_response.keys())}")
            
            # Check if we have choices
            if "choices" in json_response and len(json_response["choices"]) > 0:
                choice = json_response["choices"][0]
                print("\n--- First Choice Structure ---")
                if verbose:
                    print(json.dumps(choice, indent=2))
                else:
                    print(f"Choice keys: {list(choice.keys())}")
                
                # Extract the content
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    print("\n--- Extracted Content ---")
                    print(content)
                    
                    # Try to parse as JSON
                    try:
                        content_json = json.loads(content)
                        print("\n--- Content as JSON ---")
                        print(json.dumps(content_json, indent=2))
                    except json.JSONDecodeError:
                        print("\nContent is not valid JSON")
                    
                    print("\nTest SUCCESSFUL: Content extracted correctly")
                    return True
                else:
                    print("\nERROR: Could not find 'message.content' in the response choice")
            else:
                print("\nERROR: No 'choices' found in the response")
            
            print("\nRaw response saved to:", output_file)
            return False
            
        except json.JSONDecodeError as e:
            print(f"\nERROR: Failed to parse response as JSON: {e}")
            print("Raw response saved to:", output_file)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: curl command failed: {e}")
        return False
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenRouter API connection")
    parser.add_argument("--api-key", help="OpenRouter API key")
    parser.add_argument("--model", help="Model to test (default from config or anthropic/claude-3-opus-20240229)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    test_openrouter(args.api_key, args.model, args.verbose)
