#HOW TO RUN  python3 .\q_learning_file.py .\test_code.txt
import requests
import json
import sys

OLLAMA_API = "http://localhost:11435/api/generate"
MEMORY_FILE = "q_memory.json"

# Ensure filename is provided
if len(sys.argv) < 2:
    print("❌ Error: Please provide a filename as an argument.")
    sys.exit(1)

file_name = sys.argv[1]

# Read prompt from file
try:
    with open(file_name, "r", encoding="utf-8") as f:
        user_prompt = f.read().strip()
except FileNotFoundError:
    print(f"❌ Error: File '{file_name}' not found.")
    sys.exit(1)

# Initialize Q-learning memory
q_table = {}

def get_reward(response_text, expected_keywords):
    """Assign rewards based on presence of expected keywords."""
    return 1 if all(keyword.lower() in response_text.lower() for keyword in expected_keywords) else -1

attempts = 0
max_attempts = 3
learning_rate = 0.7  
discount_factor = 0.9  

# Load previous memory
try:
    with open(MEMORY_FILE, "r") as f:
        q_table = json.load(f)
except FileNotFoundError:
    q_table = {}

expected_keywords = user_prompt.split()  # Use words from prompt to check relevance

while attempts < max_attempts:
    response = requests.post(OLLAMA_API, json={"model": "deepseek-r1", "prompt": user_prompt, "stream": False})
    response_json = response.json()
    
    response_text = response_json.get("response", "")
    reward = get_reward(response_text, expected_keywords)

    # Update Q-table (learning)
    q_table[attempts] = q_table.get(attempts, 0) + learning_rate * (reward + discount_factor * max(q_table.values(), default=0) - q_table.get(attempts, 0))

    if reward > 0:
        print(f"✅ Correct Response on Attempt {attempts + 1}:")
        print(response_text)
        break  
    else:
        print(f"❌ Incorrect Response on Attempt {attempts + 1}, Retrying...\n")
        attempts += 1

if attempts == max_attempts:
    print("⚠️ Max attempts reached. Last response:")
    print(response_text)
