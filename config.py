#!/usr/bin/env python3

import os

# Configuration settings for the agent system
CONFIG = {
    "output_dir": "./agent_outputs",
    "memory_dir": "./agent_memory",
    "default_model": "deepseek/deepseek-chat:free",
    "api_key": "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",
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "memory_db": "agent_memory.db"
}

# Ensure output directories exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["memory_dir"], exist_ok=True)