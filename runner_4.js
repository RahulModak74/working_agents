#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const { existsSync, mkdirSync } = require('fs');
const glob = require('glob');
const util = require('util');
const globPromise = util.promisify(glob);

// Import the call_api function from call_api module
let callApi;
try {
  const { call_api } = require('./call_api');
  callApi = call_api;
} catch (error) {
  console.error("Warning: Could not import call_api module. API calls will fail:", error);
  callApi = async () => { throw new Error("call_api module not available"); };
}

// Ensure the main directory is in the path
const currentDir = path.dirname(require.main ? require.main.filename : __filename);

// Load config or create default
let CONFIG;
try {
  CONFIG = require('./config');
} catch (error) {
  // Create default config if not available
  CONFIG = {
    api_key: process.env.API_KEY || "",
    endpoint: "https://api.anthropic.com/v1/messages",
    default_model: "claude-3-opus-20240229",
    output_dir: path.join(currentDir, "outputs")
  };
  // Ensure output directory exists
  if (!existsSync(CONFIG.output_dir)) {
    mkdirSync(CONFIG.output_dir, { recursive: true });
  }
}

// Global tool registry to store all available tools from adapters
const GLOBAL_TOOL_REGISTRY = {};

// Function to check if an object is hashable (can be used as a dictionary key)
function isHashable(obj) {
  try {
    // In JavaScript, only primitive types and Symbols are truly "hashable"
    return (
      obj === null || 
      typeof obj === 'undefined' || 
      typeof obj === 'string' || 
      typeof obj === 'number' || 
      typeof obj === 'boolean' ||
      typeof obj === 'symbol'
    );
  } catch (error) {
    return false;
  }
}

async function loadAllToolAdapters() {
  // Find adapters in current directory and subdirectories
  try {
    const adapterFiles = await globPromise(path.join(currentDir, "*_adapter.js"));
    const subDirAdapterFiles = await globPromise(path.join(currentDir, "*", "*_adapter.js"));
    const allAdapterFiles = [...adapterFiles, ...subDirAdapterFiles];
    
    console.log(`Found ${allAdapterFiles.length} adapter files: ${allAdapterFiles.map(f => path.basename(f))}`);
    
    for (const adapterFile of allAdapterFiles) {
      const adapterName = path.basename(adapterFile);
      const moduleName = path.basename(adapterFile, '.js');
      
      try {
        // Dynamically import the module
        const adapterModule = require(adapterFile);
        
        // Check if the module has a TOOL_REGISTRY
        if (adapterModule.TOOL_REGISTRY) {
          // Modify the tool registry to use flexible handler functions
          const flexibleRegistry = {};
          for (const [toolId, toolHandler] of Object.entries(adapterModule.TOOL_REGISTRY)) {
            // Wrap the existing handler with a flexible function
            flexibleRegistry[toolId] = createFlexibleHandler(toolHandler);
          }
          
          console.log(`Loaded tool adapter: ${moduleName} with ${Object.keys(flexibleRegistry).length} tools`);
          
          // Update the global registry with flexible handlers
          Object.assign(GLOBAL_TOOL_REGISTRY, flexibleRegistry);
        }
        
        // Check if the module has an execute_tool function
        if (adapterModule.execute_tool) {
          console.log(`Loaded execute_tool from: ${moduleName}`);
        }
      } catch (error) {
        console.error(`Error loading adapter ${adapterName}:`, error);
      }
    }
  } catch (error) {
    console.error("Error finding adapter files:", error);
  }
}

function createFlexibleHandler(handler) {
  return async function flexibleHandler(kwargs) {
    try {
      // Try to call the original handler with all provided kwargs
      return await handler(kwargs);
    } catch (error) {
      try {
        // If that fails, try to extract the most relevant parameters
        // In JS we don't have inspect.signature like in Python, so we'll have to guess
        // based on the function's toString() or just pass the full kwargs object
        return await handler(kwargs);
      } catch (e) {
        return { error: `Tool execution failed: ${e.toString()}` };
      }
    }
  };
}

async function executeTool(toolId, kwargs) {
  if (!GLOBAL_TOOL_REGISTRY[toolId]) {
    return { error: `Unknown tool: ${toolId}` };
  }
  
  try {
    const handler = GLOBAL_TOOL_REGISTRY[toolId];
    return await handler(kwargs);
  } catch (error) {
    console.error(`Error executing tool ${toolId}:`, error);
    return { error: error.toString() };
  }
}

function extractJsonFromText(text) {
  // Try parsing the entire content as JSON first
  try {
    return JSON.parse(text);
  } catch (error) {
    // Continue with other extraction methods
  }
  
  // Try to find JSON within code blocks
  const jsonPattern = /```(?:json)?\s*([\s\S]*?)\s*```/;
  const match = text.match(jsonPattern);
  if (match) {
    try {
      return JSON.parse(match[1]);
    } catch (error) {
      // Continue with other extraction methods
    }
  }
  
  // Try to find anything that looks like a JSON object
  const objectPattern = /({[\s\S]*?})/;
  const objMatch = text.match(objectPattern);
  if (objMatch) {
    try {
      return JSON.parse(objMatch[1]);
    } catch (error) {
      // Continue with other extraction methods
    }
  }
  
  // If all extraction attempts fail, return an error object
  return { error: "Could not extract valid JSON from response", text: text.substring(0, 500) };
}

async function runAgent(agentName, prompt, filePath = null, outputFormat = null, references = null) {
  // Determine the output type and schema
  let outputType = "text";
  let schema = null;
  let sections = null;
  
  if (outputFormat) {
    outputType = outputFormat.type || "text";
    schema = outputFormat.schema;
    sections = outputFormat.sections;
  }
  
  // Enhance prompt with format instructions
  let enhancedPrompt = prompt;
  
  // Add reference information if provided
  if (references) {
    enhancedPrompt += "\n\n### Reference Information:\n";
    for (const [refName, refContent] of Object.entries(references)) {
      enhancedPrompt += `\n#### Output from ${refName}:\n`;
      if (typeof refContent === 'object') {
        enhancedPrompt += JSON.stringify(refContent, null, 2);
      } else {
        enhancedPrompt += String(refContent);
      }
    }
  }
  
  // Add file content if provided
  if (filePath && existsSync(filePath)) {
    try {
      const fileContent = await fs.readFile(filePath, 'utf-8');
      
      // For large files, just include a limited amount
      if (fileContent.length > 10000) {
        const preview = fileContent.substring(0, 10000) + "...[content truncated]...";
        enhancedPrompt += `\n\nHere is a preview of the content of ${path.basename(filePath)}:\n\`\`\`\n${preview}\n\`\`\``;
      } else {
        enhancedPrompt += `\n\nHere is the content of ${path.basename(filePath)}:\n\`\`\`\n${fileContent}\n\`\`\``;
      }
      
      console.log(`Successfully loaded file: ${filePath}`);
    } catch (error) {
      console.warn(`Warning: Could not read file ${filePath}:`, error);
    }
  } else if (filePath) {
    console.warn(`Warning: File not found: ${filePath}`);
  }
  
  // Add explicit formatting instructions
  if (outputType === "json" && schema) {
    enhancedPrompt += "\n\n### Response Format Instructions:\n";
    enhancedPrompt += "You MUST respond with a valid JSON object exactly matching this schema:\n";
    enhancedPrompt += `\`\`\`json\n${JSON.stringify(schema, null, 2)}\n\`\`\`\n`;
    enhancedPrompt += "\nReturning properly formatted JSON is CRITICAL. Do not include any explanations or text outside the JSON object.";
  } else if (outputType === "markdown" && sections) {
    enhancedPrompt += "\n\n### Response Format Instructions:\n";
    enhancedPrompt += "You MUST format your response as a Markdown document containing these exact sections:\n\n";
    for (const section of sections) {
      enhancedPrompt += `# ${section}\n\n`;
    }
    enhancedPrompt += "\nEnsure each section heading uses a single # character followed by the exact section name as listed above.";
  }
  
  // Build the payload
  const formatType = outputType === "json" ? "JSON" : "markdown";
  let systemMessage = `You are a specialized assistant handling ${formatType} outputs. Your responses must strictly follow the format specified in the instructions.`;
  
  // Add domain-specific additions to system message
  if (agentName.toLowerCase().includes("security") || agentName.toLowerCase().includes("threat") || prompt.toLowerCase().includes("cyber")) {
    systemMessage = "You are a cybersecurity analysis assistant. " + systemMessage;
  } else if (agentName.toLowerCase().includes("journey") || agentName.toLowerCase().includes("customer") || prompt.toLowerCase().includes("segment")) {
    systemMessage = "You are a customer journey analysis assistant. " + systemMessage;
  } else if (agentName.toLowerCase().includes("finance") || agentName.toLowerCase().includes("investment") || prompt.toLowerCase().includes("portfolio")) {
    systemMessage = "You are a financial analysis assistant. " + systemMessage;
  }
  
  // Add tool usage instructions if tools are needed
  if (enhancedPrompt.includes("You have access to these tools:")) {
    systemMessage += `
You have access to tools specified in the instructions. To use a tool, format your response like this:

I need to use the tool: $TOOL_NAME
Parameters:
{
  "param1": "value1",
  "param2": "value2"
}

Wait for the tool result before continuing.
`;
  }
  
  // Define conversation for tool usage
  const conversation = [
    { role: "system", content: systemMessage },
    { role: "user", content: enhancedPrompt }
  ];
  
  // Output file
  const outputFile = path.join(CONFIG.output_dir, `${agentName}_output.txt`);
  let finalResponse = null;
  
  // Execute the call with potential tool usage loop
  console.log(`🤖 Running agent: ${agentName}`);
  try {
    // Initial API call
    let apiResponse = await callApi(conversation, CONFIG);
    let responseContent = apiResponse;
    
    // Check if the response contains a tool usage request
    const toolUsagePattern = /I need to use the tool: ([a-zA-Z0-9_:]+)\s*\nParameters:\s*\{([^}]+)\}/s;
    let toolMatch = toolUsagePattern.exec(responseContent);
    
    // Loop for tool usage if needed
    const maxToolCalls = 5;  // Set a limit to avoid infinite loops
    let toolCalls = 0;
    
    while (toolMatch && toolCalls < maxToolCalls) {
      toolCalls++;
      
      // Extract tool name and parameters
      const toolName = toolMatch[1].trim();
      const paramsText = "{" + toolMatch[2] + "}";
      
      try {
        // Parse parameters
        const params = JSON.parse(paramsText);
        console.log(`📡 Tool call ${toolCalls}: ${toolName}`);
        
        // Execute the tool
        if (GLOBAL_TOOL_REGISTRY[toolName]) {
          const toolResult = await executeTool(toolName, params);
          const toolResultStr = JSON.stringify(toolResult, null, 2);
          
          // Add the tool interaction to the conversation
          conversation.push({ role: "assistant", content: responseContent });
          conversation.push({
            role: "user", 
            content: `Tool result for ${toolName}:\n\`\`\`json\n${toolResultStr}\n\`\`\`\n\nPlease continue based on this result.`
          });
          
          // Get response with the tool result
          apiResponse = await callApi(conversation, CONFIG);
          responseContent = apiResponse;
          
          // Check for another tool usage
          toolMatch = toolUsagePattern.exec(responseContent);
        } else {
          // Tool not found
          conversation.push({ role: "assistant", content: responseContent });
          conversation.push({
            role: "user", 
            content: `Error: Tool '${toolName}' not found. Please continue without using this tool.`
          });
          
          // Get response after tool error
          apiResponse = await callApi(conversation, CONFIG);
          responseContent = apiResponse;
          
          // Check for another tool usage
          toolMatch = toolUsagePattern.exec(responseContent);
        }
      } catch (error) {
        // Invalid JSON in parameters
        conversation.push({ role: "assistant", content: responseContent });
        conversation.push({
          role: "user", 
          content: `Error: Invalid parameter format. Parameters must be valid JSON. Please continue without using this tool.`
        });
        
        // Get response after parameter error
        apiResponse = await callApi(conversation, CONFIG);
        responseContent = apiResponse;
        
        // Check for another tool usage
        toolMatch = toolUsagePattern.exec(responseContent);
      }
    }
    
    // Final response after all tool usage
    finalResponse = responseContent;
    
    // Save the content to the output file
    await fs.writeFile(outputFile, finalResponse, 'utf-8');
    
    // Process based on output type
    if (outputType === "json") {
      const result = extractJsonFromText(finalResponse);
      if ("error" in result) {
        console.warn(`⚠️ Warning: JSON extraction failed for ${agentName}. Content: ${finalResponse.substring(0, 100)}...`);
      }
      console.log(`✅ ${agentName} completed`);
      return result;
    } else if (outputType === "markdown") {
      // Verify sections if required
      if (sections) {
        const missingSections = [];
        for (const section of sections) {
          const sectionRegex = new RegExp(`#\\s*${section.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`, 'i');
          if (!sectionRegex.test(finalResponse)) {
            missingSections.push(section);
          }
        }
        
        if (missingSections.length > 0) {
          console.warn(`Warning: Missing ${missingSections.length} required sections in markdown output`);
        }
      }
      
      console.log(`✅ ${agentName} completed`);
      return { markdown_content: finalResponse };
    } else {
      console.log(`✅ ${agentName} completed`);
      return { text_content: finalResponse };
    }
  } catch (error) {
    console.error(`❌ Error with ${agentName}:`, error);
    return { error: `Error: ${error.toString()}`, content: error.toString() };
  }
}

async function runUniversalWorkflow(workflowFile, dataFile = null) {
  // Get the directory where the workflow file is located
  const workflowDir = path.dirname(path.resolve(workflowFile));
  
  // First, load all tool adapters
  await loadAllToolAdapters();
  
  // Print available tools for debugging
  console.log(`Loaded a total of ${Object.keys(GLOBAL_TOOL_REGISTRY).length} tools from all adapters`);
  if (Object.keys(GLOBAL_TOOL_REGISTRY).length > 0) {
    console.log(`Available tools: ${Object.keys(GLOBAL_TOOL_REGISTRY).sort().join(', ')}`);
  }
  
  // Load the workflow
  const workflow = JSON.parse(await fs.readFile(workflowFile, 'utf-8'));
  
  // Store results for each agent
  const results = {};
  
  // Process each agent in the workflow
  for (const step of workflow) {
    const agentName = step.agent;
    const content = step.content || "";
    const fileParam = step.file;
    const outputFormat = step.output_format || {};
    
    // Collect references from previous agents
    const references = {};
    if (step.readFrom) {
      for (const refName of step.readFrom) {
        // Handle all reference types
        if (refName === "*") {
          // Include all previous results except this agent
          for (const [prevAgent, prevResult] of Object.entries(results)) {
            if (prevAgent !== agentName && !references[prevAgent]) {
              references[prevAgent] = prevResult;
            }
          }
        } else if (isHashable(refName) && results[refName]) {
          // The reference is to a previous result
          references[refName] = results[refName];
        } else if (typeof refName === 'object' && refName !== null) {
          // Try to extract a usable key from the dictionary
          if ('id' in refName && isHashable(refName.id) && results[refName.id]) {
            references[String(refName.id)] = results[refName.id];
          } else if ('agent' in refName && isHashable(refName.agent) && results[refName.agent]) {
            references[String(refName.agent)] = results[refName.agent];
          } else if ('name' in refName && isHashable(refName.name) && results[refName.name]) {
            references[String(refName.name)] = results[refName.name];
          } else {
            console.warn(`Warning: Could not resolve reference:`, refName);
          }
        }
      }
    }
    
    // Check if tools are required for this step
    let stepContent = content;
    if (step.tools) {
      const requiredTools = step.tools;
      const missingTools = requiredTools.filter(tool => !GLOBAL_TOOL_REGISTRY[tool]);
      
      if (missingTools.length > 0) {
        console.warn(`Warning: Missing required tools for ${agentName}: ${missingTools.join(', ')}`);
        // Add note about missing tools to the agent prompt
        stepContent += `\n\nNote: The following tools are not available: ${missingTools.join(', ')}`;
      } else {
        // Add note about available tools to the agent prompt
        stepContent += `\n\nYou have access to these tools: ${requiredTools.join(', ')}`;
      }
    }
    
    // Handle dynamic agent
    if (step.type === "dynamic") {
      const initialPrompt = step.initial_prompt || "";
      
      const result = await runAgent(
        agentName,
        initialPrompt,
        dataFile && fileParam ? dataFile : null,
        outputFormat,
        references
      );
      
      results[agentName] = result;
      
      // Determine action from result
      let actionKey = null;
      
      // Try to extract action from various possible fields
      if (typeof result === 'object' && result !== null) {
        for (const key of ["response_action", "action", "selected_focus"]) {
          if (key in result) {
            actionKey = result[key];
            break;
          }
        }
      }
      
      // Store the action
      const actionName = `${agentName}_action`;
      results[actionName] = actionKey;
      console.log(`🔍 Dynamic agent selected action: ${actionKey}`);
      
      // Check if action is valid and exists in actions
      if (actionKey && isHashable(actionKey) && step.actions && step.actions[actionKey]) {
        const action = step.actions[actionKey];
        const nextAgentName = action.agent;
        
        if (nextAgentName) {
          const actionContent = action.content || "";
          
          // Collect references for the action
          const actionRefs = {};
          if (action.readFrom) {
            for (const refName of action.readFrom) {
              if (refName === "*") {
                for (const [prevAgent, prevResult] of Object.entries(results)) {
                  if (prevAgent !== nextAgentName && !actionRefs[prevAgent]) {
                    actionRefs[prevAgent] = prevResult;
                  }
                }
              } else if (isHashable(refName) && results[refName]) {
                actionRefs[refName] = results[refName];
              } else if (typeof refName === 'object' && refName !== null) {
                console.warn(`Warning: Dictionary reference in action:`, refName);
              }
            }
          }
          
          // Run the next agent
          const actionResult = await runAgent(
            nextAgentName,
            actionContent,
            dataFile && action.file ? dataFile : null,
            action.output_format,
            actionRefs
          );
          
          results[nextAgentName] = actionResult;
        }
      } else if (actionKey) {
        // Handle non-hashable action_key or action_key not in actions
        if (!isHashable(actionKey)) {
          console.warn(`Warning: Dynamic agent selected a non-hashable action: ${typeof actionKey}`);
        } else {
          console.warn(`Warning: Dynamic agent selected invalid action: ${actionKey}`);
        }
      }
    } else {
      // Standard agent execution
      const result = await runAgent(
        agentName,
        stepContent,
        dataFile && fileParam ? dataFile : null,
        outputFormat,
        references
      );
      
      results[agentName] = result;
    }
  }
  
  // Save the complete results
  const outputFile = path.join(CONFIG.output_dir, "workflow_results.json");
  try {
    await fs.writeFile(outputFile, JSON.stringify(results, null, 2), 'utf-8');
    console.log(`Complete analysis saved to ${outputFile}`);
  } catch (error) {
    console.error(`Error saving results:`, error);
  }
  
  return results;
}

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  
  if (args.length > 0 && args[0] === "--workflow") {
    if (args.length < 2) {
      console.log("Usage: node runner.js --workflow <workflow_file> [data_file]");
      process.exit(1);
    }
    
    const workflowFile = args[1];
    
    if (!existsSync(workflowFile)) {
      console.error(`Workflow file not found: ${workflowFile}`);
      process.exit(1);
    }
    
    let dataFile = null;
    if (args.length > 2) {
      dataFile = args[2];
      if (!existsSync(dataFile)) {
        console.error(`Data file not found: ${dataFile}`);
        process.exit(1);
      }
    }
    
    // Run the workflow
    console.log(`Executing workflow: ${workflowFile}`);
    const results = await runUniversalWorkflow(workflowFile, dataFile);
    
    // Print a summary of the results
    console.log("\nWorkflow completed with results:");
    for (const [agent, result] of Object.entries(results)) {
      if (agent.includes("_action")) {
        console.log(`\n=== ${agent} ===`);
        console.log(result);
      } else if (typeof result === 'object' && result !== null && "error" in result) {
        console.log(`\n=== ${agent} ===`);
        console.log(`Error: ${result.error}`);
      } else {
        console.log(`\n=== ${agent} ===`);
        if (typeof result === 'object' && result !== null) {
          console.log("✓ Success");
        } else {
          console.log(result);
        }
      }
    }
  } else {
    // Start the interactive CLI if available
    try {
      const { AgentShell } = require('./cli');
      new AgentShell().cmdloop();
    } catch (error) {
      console.error("Error: Could not import AgentShell from cli");
      console.log("Usage: node runner.js --workflow <workflow_file> [data_file]");
      process.exit(1);
    }
  }
}

// Export functions for use in other modules
module.exports = {
  runAgent,
  runUniversalWorkflow,
  executeTool,
  loadAllToolAdapters,
  GLOBAL_TOOL_REGISTRY
};

// Run main function if this script is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error("Error in main:", error);
    process.exit(1);
  });
}
