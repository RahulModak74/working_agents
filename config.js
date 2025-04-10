/**
 * Configuration settings for the agent system
 */
const fs = require('fs');
const path = require('path');

// Configuration settings
const CONFIG = {
  "output_dir": "./agent_outputs",
  "memory_dir": "./agent_memory",
  "default_model": "openrouter/quasar-alpha",
  "api_key": "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",
  "endpoint": "https://openrouter.ai/api/v1/chat/completions",
  "memory_db": "agent_memory.db"
};

// Ensure output directories exist
if (!fs.existsSync(CONFIG.output_dir)) {
  fs.mkdirSync(CONFIG.output_dir, { recursive: true });
}

if (!fs.existsSync(CONFIG.memory_dir)) {
  fs.mkdirSync(CONFIG.memory_dir, { recursive: true });
}

module.exports = CONFIG;
