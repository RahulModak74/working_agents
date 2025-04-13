import subprocess
import json

def call_api(conversation, config):
    """
    Call the API with the given conversation and configuration.
    Parameters:
        conversation (list): The list of messages to send to the API.
        config (dict): API configuration (endpoint, api_key, default_model).
    Returns:
        str: The response content from the API.
    """
    payload = {
        "model": config["default_model"],
        "messages": conversation
    }

    # Safely encode the payload to JSON
    payload_str = json.dumps(payload).replace("'", "'\\''")

    curl_command = f"""curl {config["endpoint"]} \\
      -H "Authorization: Bearer {config["api_key"]}" \\
      -H "Content-Type: application/json" \\
      -d '{payload_str}'"""

    output_file = "temp_output.json"
    try:
        subprocess.run(curl_command + f" -o {output_file}", shell=True, check=True)
        
        with open(output_file, 'r', encoding='utf-8') as f:
            response = f.read()

        json_response = json.loads(response)

        content = json_response.get('content', '') or \
                  json_response.get('choices', [{}])[0].get('message', {}).get('content', '')

        return content

    except Exception as e:
        print(f"Error calling API: {e}")
        return f"Error: {str(e)}"
