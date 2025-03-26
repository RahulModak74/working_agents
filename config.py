#!/usr/bin/env python3

import os

# Configuration settings for the agent system
CONFIG = {
    "output_dir": "./agent_outputs",
    "memory_dir": "./agent_memory",
    "default_model": "deepseek/deepseek-chat:free",
    "api_key": "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxx",
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "memory_db": "agent_memory.db"
}

# Ensure output directories exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["memory_dir"], exist_ok=True)
