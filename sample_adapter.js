/**
 * Sample tool adapter for the Node.js runner
 */
const fs = require('fs').promises;
const path = require('path');

/**
 * Read a file and return its contents
 */
async function readFileTool(params) {
  try {
    const filePath = params.file_path;
    if (!filePath) {
      return { error: "No file path provided" };
    }
    
    const content = await fs.readFile(filePath, 'utf-8');
    return { 
      success: true, 
      content: content.length > 1000 ? 
        content.substring(0, 1000) + "... [truncated]" : 
        content 
    };
  } catch (error) {
    return { error: `Failed to read file: ${error.message}` };
  }
}

/**
 * Get the current date and time
 */
async function getCurrentDateTool() {
  return {
    date: new Date().toISOString(),
    timestamp: Date.now()
  };
}

/**
 * Analyze text using basic NLP techniques
 */
async function analyzeTextTool(params) {
  const text = params.text || "";
  
  if (!text) {
    return { error: "No text provided for analysis" };
  }
  
  // Simple word count and stats
  const words = text.split(/\s+/).filter(word => word.length > 0);
  const sentences = text.split(/[.!?]+/).filter(sentence => sentence.trim().length > 0);
  
  return {
    word_count: words.length,
    sentence_count: sentences.length,
    avg_word_length: words.length > 0 ? 
      words.join('').length / words.length : 0,
    avg_sentence_length: sentences.length > 0 ? 
      words.length / sentences.length : 0
  };
}

// Export the tool registry
const TOOL_REGISTRY = {
  "read_file": readFileTool,
  "current_date": getCurrentDateTool,
  "analyze_text": analyzeTextTool
};

/**
 * Generic tool execution function
 */
async function execute_tool(tool_id, params = {}) {
  if (TOOL_REGISTRY[tool_id]) {
    return await TOOL_REGISTRY[tool_id](params);
  } else {
    return { error: `Unknown tool ID: ${tool_id}` };
  }
}

module.exports = {
  TOOL_REGISTRY,
  execute_tool
};
