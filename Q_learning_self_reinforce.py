
# python3 .\q_learning_ollama_analysis.py "Pl code python script to  fit pymc3 regression"  (How to run)
import requests
import sys
#Normal ollama is on port 11434 .. I hv used port 11435 for other reasons
OLLAMA_API = "http://localhost:11435/api/generate"

# Ensure question is provided via command-line
if len(sys.argv) < 2:
    print("❌ Error: Please provide a prompt as an argument.")
    sys.exit(1)

user_prompt = sys.argv[1]

# Initialize Q-learning memory
q_table = {}

def get_reward(response_text, expected_keywords):
    """Assign rewards based on presence of expected keywords."""
    return 1 if all(keyword.lower() in response_text.lower() for keyword in expected_keywords) else -1

attempts = 0
max_attempts = 3
learning_rate = 0.7  
discount_factor = 0.9  
MEMORY_FILE = "q_memory.json"

# Load previous memory
try:
    with open(MEMORY_FILE, "r") as f:
        q_table = json.load(f)
except FileNotFoundError:
    q_table = {}

expected_keywords = user_prompt.split()  # Use words from the prompt to check relevance

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
