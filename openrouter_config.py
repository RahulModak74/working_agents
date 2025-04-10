#!/usr/bin/env python3

import os

# Configuration settings for the agent system
CONFIG = {
    "output_dir": "./agent_outputs",
    "memory_dir": "./agent_memory",
#    "default_model": "deepseek/deepseek-chat:free",
     "default_model": "openrouter/quasar-alpha",
    "api_key": "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",
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "memory_db": "agent_memory.db"
}

# Ensure output directories exist
os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["memory_dir"], exist_ok=True)
