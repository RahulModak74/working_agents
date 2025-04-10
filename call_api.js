const axios = require('axios');

/**
 * Calls the OpenRouter API with a conversation and config
 * @param {Array} conversation - Array of message objects with role and content
 * @param {Object} config - Configuration object containing API settings
 * @returns {Promise<string>} - The assistant's response content
 */
async function call_api(conversation, config) {
  try {
    const apiKey = config.api_key || process.env.API_KEY;
    
    if (!apiKey) {
      throw new Error("API key is required. Set it in config.js or as API_KEY environment variable.");
    }

    // Convert the conversation to the OpenRouter format
    const messages = conversation.map(msg => ({
      role: msg.role,
      content: msg.content
    }));

    const response = await axios({
      method: 'post',
      url: config.endpoint,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        'HTTP-Referer': 'https://localhost:3000', // Replace with your actual domain
        'X-Title': 'Agent Runner'
      },
      data: {
        model: config.default_model,
        messages: messages,
        max_tokens: 4000
      }
    });

    // Extract the assistant's response content from OpenRouter format
    if (response.data && response.data.choices && response.data.choices.length > 0) {
      return response.data.choices[0].message.content;
    } else {
      throw new Error("Unexpected API response format");
    }
  } catch (error) {
    console.error("API call failed:", error.response?.data || error.message);
    throw new Error(`API call failed: ${error.message}`);
  }
}

module.exports = { call_api };
