#!/usr/bin/env python3
"""
Cross-Platform Workflow Runner for Microsoft Intelligence Analysis
Compatible with both Windows and Linux - All dependencies included
"""

import asyncio
import aiohttp
import json
import time
import os
import sys
import logging
import re
import importlib.util
import inspect
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from functools import wraps

# Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("workflow_runner")

# Configuration Management
def get_config():
    """Get configuration with multiple fallbacks"""
    config_files = ['config.py', 'openrouter_config.py', 'ollama_config.py']
    
    for config_file in config_files:
        if Path(config_file).exists():
            try:
                spec = importlib.util.spec_from_file_location("config", config_file)
                if spec and spec.loader:
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    return config_module.CONFIG
            except Exception as e:
                logger.debug(f"Failed to load {config_file}: {e}")
    
    # Default config
    return {
        "output_dir": str(Path.cwd() / "agent_outputs"),
        "default_model": "deepseek/deepseek-chat:free",
        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "timeout": 300
    }

# Utility Functions
def extract_json_from_text(text: str) -> Dict[str, Any]:
    """Extract JSON from text response"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    object_pattern = r'({[\s\S]*?})'
    match = re.search(object_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return {"error": "Could not extract valid JSON", "text": text[:500]}

def extract_tool_calls(response_content: str) -> List[Dict[str, Any]]:
    """Extract tool calls from response"""
    tool_usage_pattern = r"I need to use the tool: ([a-zA-Z0-9_:]+)\s*\nParameters:\s*\{([^}]+)\}"
    tool_calls = []
    
    matches = re.finditer(tool_usage_pattern, response_content, re.DOTALL)
    for match in matches:
        tool_name = match.group(1).strip()
        params_text = "{" + match.group(2) + "}"
        try:
            params = json.loads(params_text)
            tool_calls.append({
                "tool_name": tool_name,
                "params": params,
                "full_text": match.group(0)
            })
        except json.JSONDecodeError:
            continue
    
    return tool_calls

# Tool Manager Implementation
class ToolManager:
    def __init__(self):
        self.tools = {}
        self.imported_modules = set()
        self.excluded_files = {
            'agent_runner.py', 'tool_manager.py', 'utils.py', 'config.py', 'call_api.py',
            '__init__.py', 'cli.py', 'async_executor.py', 'async_framework_main.py',
            'workflow_executor.py', 'workflow_state.py', 'enhanced_agent_runner.py'
        }
        self.excluded_functions = {
            'main', '__init__', 'setup', 'teardown', 'test_', 'debug_'
        }
    
    def discover_tools(self, directories: List[str] = None) -> int:
        """Auto-discover tools from Python modules"""
        current_dir = Path(__file__).parent
        if directories is None:
            directories = [current_dir]
            
            component_dir = current_dir / "COMPONENT"
            if component_dir.exists():
                directories.append(component_dir)
            
            for item in current_dir.iterdir():
                if (item.is_dir() and 
                    not item.name.startswith('.') and 
                    item.name not in ['__pycache__', 'COMPONENT', 'agent_outputs']):
                    directories.append(item)
        
        total_tools = 0
        logger.info(f"Scanning directories: {[str(d) for d in directories]}")
        
        for directory in directories:
            directory = Path(directory)
            if not directory.exists():
                continue
                
            python_files = list(directory.glob("*.py"))
            logger.info(f"Found {len(python_files)} Python files in {directory}")
            
            for py_file in python_files:
                if py_file.name in self.excluded_files:
                    continue
                
                module_name = py_file.stem
                if module_name in self.imported_modules:
                    continue
                
                try:
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None or spec.loader is None:
                        continue
                        
                    module = importlib.util.module_from_spec(spec)
                    
                    # Add directory to path temporarily
                    str_dir = str(py_file.parent)
                    if str_dir not in sys.path:
                        sys.path.insert(0, str_dir)
                        added_to_path = True
                    else:
                        added_to_path = False
                    
                    try:
                        spec.loader.exec_module(module)
                        self.imported_modules.add(module_name)
                        
                        prefix = getattr(module, 'TOOL_NAMESPACE', module_name)
                        if prefix.endswith("_adapter"):
                            prefix = prefix[:-8]
                        
                        # Register from TOOL_REGISTRY
                        registry_tools = 0
                        if hasattr(module, 'TOOL_REGISTRY'):
                            tool_registry = getattr(module, 'TOOL_REGISTRY')
                            if isinstance(tool_registry, dict):
                                for tool_id, tool_handler in tool_registry.items():
                                    full_tool_id = f"{prefix}:{tool_id}" if ':' not in tool_id else tool_id
                                    self.register_tool(full_tool_id, tool_handler)
                                    registry_tools += 1
                                    total_tools += 1
                        
                        # Auto-discover functions
                        auto_tools = self._discover_module_tools(module, prefix)
                        total_tools += len(auto_tools)
                        
                        logger.info(f"Imported {module_name}: {registry_tools} registry + {len(auto_tools)} auto tools")
                        
                    finally:
                        if added_to_path and str_dir in sys.path:
                            sys.path.remove(str_dir)
                    
                except Exception as e:
                    logger.error(f"Error importing {module_name}: {e}")
                    continue
        
        logger.info(f"🔧 Total tools discovered: {total_tools}")
        return total_tools
    
    def _discover_module_tools(self, module, prefix: str) -> List[str]:
        """Discover tools from module functions"""
        discovered_tools = []
        
        for name, obj in inspect.getmembers(module):
            if name.startswith('_') or not inspect.isfunction(obj):
                continue
            
            if any(name.startswith(excluded) for excluded in self.excluded_functions):
                continue
                
            if hasattr(module, 'TOOL_REGISTRY'):
                tool_registry = getattr(module, 'TOOL_REGISTRY')
                if isinstance(tool_registry, dict) and any(handler == obj for handler in tool_registry.values()):
                    continue
            
            if obj.__doc__ and obj.__doc__.strip():
                tool_id = f"{prefix}:{name}"
                self.register_tool(tool_id, obj)
                discovered_tools.append(tool_id)
        
        return discovered_tools
    
    def register_tool(self, tool_id: str, handler: Callable) -> None:
        """Register a tool with flexible parameter handling"""
        @wraps(handler)
        def flexible_handler(**kwargs):
            try:
                sig = inspect.signature(handler)
                
                if any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
                    return handler(**kwargs)
                
                filtered_kwargs = {}
                for param_name, param in sig.parameters.items():
                    if param_name in kwargs:
                        filtered_kwargs[param_name] = kwargs[param_name]
                    elif param.default == param.empty and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                        return {"error": f"Required parameter '{param_name}' missing"}
                
                return handler(**filtered_kwargs)
                
            except Exception as e:
                return {"error": f"Tool execution failed: {str(e)}"}
        
        self.tools[tool_id] = flexible_handler
    
    def execute_tool(self, tool_id: str, **kwargs) -> Any:
        """Execute a registered tool"""
        if tool_id not in self.tools:
            return {"error": f"Unknown tool: {tool_id}"}
        
        try:
            logger.info(f"Executing tool {tool_id}")
            result = self.tools[tool_id](**kwargs)
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_tools(self) -> List[str]:
        """Get all registered tool IDs"""
        return sorted(list(self.tools.keys()))
    
    def is_tool_available(self, tool_id: str) -> bool:
        """Check if tool is available"""
        return tool_id in self.tools

# Global tool manager instance
tool_manager = ToolManager()

# Agent Task Definition
@dataclass
class AgentTask:
    agent_name: str
    prompt: str
    file_path: Optional[str] = None
    output_format: Optional[Dict[str, Any]] = None
    references: Optional[Dict[str, Any]] = None
    required_tools: List[str] = None
    dependencies: Set[str] = None
    step_index: int = 0

# Async Workflow Executor
class AsyncToolIntegratedExecutor:
    def __init__(self, config: Dict[str, Any], max_concurrent: int = 8):
        self.config = config
        self.max_concurrent = max_concurrent
        self.rate_limiter = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.connector = None
        self.results = {}
        self.completed_agents = set()
        self.failed_agents = set()
        self.request_times = deque()
        self.rate_limit = 2.0
        
    async def __aenter__(self):
        """Initialize HTTP session"""
        self.connector = aiohttp.TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=10,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=300, connect=30)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up HTTP session"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def _rate_limit(self):
        """Rate limiting implementation"""
        async with self.rate_limiter:
            now = time.time()
            self.request_times = deque([t for t in self.request_times if now - t < 1.0])
            
            if len(self.request_times) >= self.rate_limit:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(now)
    
    async def _call_api_async(self, conversation: List[Dict], agent_name: str = "unknown") -> str:
        """Async API call with retries"""
        await self._rate_limit()
        
        payload = {
            "model": self.config["default_model"],
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        headers = {}
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        
        for attempt in range(3):
            try:
                async with self.session.post(
                    self.config["endpoint"],
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = (data.get('content', '') or 
                                 data.get('choices', [{}])[0].get('message', {}).get('content', ''))
                        return content
                    elif response.status == 429:
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited for {agent_name}, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await response.text()
                        logger.error(f"API error {response.status} for {agent_name}: {error_text}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {agent_name} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"API error for {agent_name}: {e}")
            
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
        
        return f"Error: API request failed for {agent_name}"
    
    async def _execute_agent_with_tools(self, task: AgentTask) -> Dict[str, Any]:
        """Execute agent with tool support"""
        try:
            enhanced_prompt = task.prompt
            
            # Add tool information
            if task.required_tools:
                available_tools = []
                for tool_id in task.required_tools:
                    if tool_manager.is_tool_available(tool_id):
                        available_tools.append(tool_id)
                
                if available_tools:
                    enhanced_prompt += f"\n\nAvailable tools: {', '.join(available_tools)}"
                    enhanced_prompt += """
To use a tool, format your response like this:

I need to use the tool: TOOL_NAME
Parameters:
{
  "param1": "value1",
  "param2": "value2"
}

Wait for the tool result before continuing.
"""
            
            # Add references
            if task.references:
                enhanced_prompt += "\n\n### Reference Information:\n"
                for ref_name, ref_content in task.references.items():
                    enhanced_prompt += f"\n#### {ref_name}:\n"
                    if isinstance(ref_content, dict):
                        enhanced_prompt += json.dumps(ref_content, indent=2)
                    else:
                        enhanced_prompt += str(ref_content)
            
            # Add file content
            if task.file_path and Path(task.file_path).exists():
                try:
                    file_content = Path(task.file_path).read_text(encoding='utf-8')
                    if len(file_content) > 10000:
                        file_content = file_content[:10000]
                    enhanced_prompt += f"\n\nFile content:\n```\n{file_content}\n```"
                except Exception as e:
                    logger.warning(f"Could not read file {task.file_path}: {e}")
            
            # Add format instructions
            if task.output_format and task.output_format.get("type") == "json":
                schema = task.output_format.get("schema")
                if schema:
                    enhanced_prompt += "\n\n### Response Format:\n"
                    enhanced_prompt += "Return valid JSON matching this schema:\n"
                    enhanced_prompt += f"```json\n{json.dumps(schema, indent=2)}\n```"
            
            # Build conversation
            conversation = [
                {"role": "system", "content": "You are a specialized assistant. Follow instructions precisely."},
                {"role": "user", "content": enhanced_prompt}
            ]
            
            # Make API call
            response_content = await self._call_api_async(conversation, task.agent_name)
            
            # Handle tool calls
            if "I need to use the tool:" in response_content:
                tool_calls = extract_tool_calls(response_content)
                
                if tool_calls:
                    for tool_call in tool_calls[:3]:  # Limit to 3 tools
                        tool_name = tool_call["tool_name"]
                        params = tool_call["params"]
                        
                        logger.info(f"Executing tool: {tool_name}")
                        tool_result = tool_manager.execute_tool(tool_name, **params)
                        tool_result_str = json.dumps(tool_result, indent=2)
                        
                        # Replace tool call with result
                        response_content = response_content.replace(
                            tool_call["full_text"],
                            f"Tool result for {tool_name}:\n```json\n{tool_result_str}\n```"
                        )
                    
                    # Get final response
                    final_prompt = f"Based on the tool results:\n\n{response_content}\n\nProvide your final analysis."
                    conversation.append({"role": "assistant", "content": response_content})
                    conversation.append({"role": "user", "content": final_prompt})
                    
                    response_content = await self._call_api_async(conversation, task.agent_name)
            
            # Process response
            if task.output_format and task.output_format.get("type") == "json":
                result = extract_json_from_text(response_content)
            else:
                result = {"content": response_content}
            
            logger.info(f"[OK] Agent {task.agent_name} completed")
            
            # Save individual agent output
            try:
                output_dir = Path(self.config.get("output_dir", "./agent_outputs"))
                output_dir.mkdir(exist_ok=True)
                
                # Save agent result to file
                agent_output_file = output_dir / f"{task.agent_name}_output.json"
                with open(agent_output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "agent_name": task.agent_name,
                        "timestamp": datetime.now().isoformat(),
                        "result": result,
                        "prompt": task.prompt[:200] + "..." if len(task.prompt) > 200 else task.prompt
                    }, f, indent=2, default=str)
                
                logger.info(f"[FILE] Agent output saved to: {agent_output_file}")
                
            except Exception as e:
                logger.warning(f"[WARNING] Could not save agent output: {e}")
            
            return result
            
        except Exception as e:
            error_result = {"error": f"Agent execution failed: {str(e)}"}
            logger.error(f"[ERROR] Agent {task.agent_name} failed: {e}")
            return error_result
    
    
    def create_task_from_step(self, step: Dict[str, Any], step_index: int) -> AgentTask:
        """Create AgentTask from workflow step"""
        return AgentTask(
            agent_name=step.get("agent", ""),
            prompt=step.get("content", ""),
            file_path=step.get("file"),
            output_format=step.get("output_format"),
            required_tools=step.get("tools", []),
            step_index=step_index
        )
    
    def _build_dependency_graph(self, workflow: List[Dict]) -> Dict[str, Set[str]]:
        """Build dependency graph"""
        dependencies = {}
        
        for step in workflow:
            agent_name = step.get("agent", "")
            deps = set()
            
            read_from = step.get("readFrom", [])
            for ref in read_from:
                if isinstance(ref, str) and ref != "*":
                    deps.add(ref)
                elif ref == "*":
                    for prev_step in workflow:
                        if prev_step.get("agent") == agent_name:
                            break
                        prev_agent = prev_step.get("agent")
                        if prev_agent:
                            deps.add(prev_agent)
            
            dependencies[agent_name] = deps
        
        return dependencies
    
    def _create_execution_batches(self, workflow: List[Dict]) -> List[List[AgentTask]]:
        """Create execution batches based on dependencies"""
        dependencies = self._build_dependency_graph(workflow)
        
        tasks = {}
        for step_index, step in enumerate(workflow):
            agent_name = step.get("agent", "")
            if agent_name:
                task = self.create_task_from_step(step, step_index)
                task.dependencies = dependencies.get(agent_name, set())
                tasks[agent_name] = task
        
        batches = []
        completed = set()
        remaining = set(tasks.keys())
        
        while remaining:
            ready = []
            for agent_name in remaining:
                task = tasks[agent_name]
                if task.dependencies.issubset(completed):
                    ready.append(task)
            
            if not ready:
                ready = [tasks[name] for name in list(remaining)[:self.max_concurrent]]
            
            batch = ready[:self.max_concurrent]
            batches.append(batch)
            
            for task in batch:
                completed.add(task.agent_name)
                remaining.discard(task.agent_name)
        
        return batches
    
    async def execute_workflow_with_tools(self, workflow: List[Dict], data_file: str = None) -> Dict[str, Any]:
        """Execute workflow with tool integration"""
        logger.info(f"Starting workflow execution: {len(workflow)} agents")
        
        batches = self._create_execution_batches(workflow)
        logger.info(f"Created {len(batches)} execution batches")
        
        for batch_idx, batch in enumerate(batches):
            batch_start_time = time.time()
            agent_names = [task.agent_name for task in batch]
            logger.info(f"Batch {batch_idx + 1}/{len(batches)}: {agent_names}")
            
            # Update references
            for task in batch:
                refs = {}
                for dep in task.dependencies:
                    if dep in self.results:
                        refs[dep] = self.results[dep]
                task.references = refs
            
            # Execute batch
            batch_tasks = [self._execute_agent_with_tools(task) for task in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Store results
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.results[task.agent_name] = {"error": str(result)}
                    self.failed_agents.add(task.agent_name)
                else:
                    self.results[task.agent_name] = result
                    self.completed_agents.add(task.agent_name)
            
            batch_duration = time.time() - batch_start_time
            logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.1f}s")
        
        return {
            "results": self.results,
            "completed_count": len(self.completed_agents),
            "failed_count": len(self.failed_agents),
            "total_count": len(workflow)
        }

# Workflow Definitions
def create_simplified_microsoft_workflow():
    """Create simplified Microsoft workflow"""
    workflow = {
        "workflow_name": "Microsoft_Intelligence_Simplified",
        "description": "Simplified Microsoft competitive analysis with real agents",
        "steps": [
            {
                "agent": "microsoft_stock_analyzer",
                "content": "Research Microsoft's current stock performance, market cap, and recent earnings. Find latest financial metrics and analyst ratings for MSFT stock.",
                "tools": ["research:combined_search"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "stock_price": "string",
                        "market_cap": "string",
                        "recent_earnings": "object",
                        "analyst_ratings": "array"
                    }
                }
            },
            {
                "agent": "azure_market_researcher", 
                "content": "Research Microsoft Azure's current market share in cloud computing. Compare with Amazon AWS and Google Cloud Platform market positions.",
                "tools": ["research:combined_search", "cite:sources"],
                "readFrom": ["microsoft_stock_analyzer"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "azure_market_share": "string",
                        "aws_comparison": "object",
                        "gcp_comparison": "object",
                        "growth_trends": "array"
                    }
                }
            },
            {
                "agent": "office365_competitive_analyst",
                "content": "Analyze Microsoft Office 365 competitive position against Google Workspace. Research user adoption rates and feature comparisons.",
                "tools": ["research:combined_search", "research:analyze_content"],
                "readFrom": ["microsoft_stock_analyzer"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "office365_users": "string",
                        "google_workspace_threat": "object",
                        "feature_advantages": "array",
                        "market_trends": "array"
                    }
                }
            },
            {
                "agent": "microsoft_ai_strategy_analyst",
                "content": "Research Microsoft's AI strategy including Copilot, OpenAI partnership, and competition with Google's AI offerings.",
                "tools": ["research:combined_search", "research:generate_summary"],
                "readFrom": ["azure_market_researcher"],
                "output_format": {
                    "type": "json", 
                    "schema": {
                        "copilot_adoption": "string",
                        "openai_partnership": "object",
                        "google_ai_competition": "object",
                        "ai_revenue_impact": "string"
                    }
                }
            },
            {
                "agent": "google_competitive_threat_analyzer",
                "content": "Analyze Google as Microsoft's primary competitor. Research Google Cloud, Workspace, and AI initiatives that threaten Microsoft's market position.",
                "tools": ["research:combined_search", "research:analyze_content"],
                "readFrom": ["azure_market_researcher", "office365_competitive_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "google_cloud_growth": "object",
                        "workspace_vs_office365": "object",
                        "google_ai_threat": "object",
                        "strategic_moves": "array"
                    }
                }
            },
            {
                "agent": "amazon_aws_threat_analyzer",
                "content": "Analyze Amazon AWS as Microsoft Azure's biggest competitor. Research AWS market dominance and competitive strategies.",
                "tools": ["research:combined_search", "cite:sources"],
                "readFrom": ["azure_market_researcher"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "aws_market_dominance": "object",
                        "azure_vs_aws_features": "object",
                        "pricing_competition": "object",
                        "enterprise_adoption": "array"
                    }
                }
            },
            {
                "agent": "microsoft_financial_trend_analyzer",
                "content": "Analyze Microsoft's financial trends, revenue growth by segment, and future projections based on competitive landscape.",
                "tools": ["research:combined_search", "research:generate_summary"],
                "readFrom": ["microsoft_stock_analyzer", "azure_market_researcher", "office365_competitive_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "revenue_by_segment": "object",
                        "growth_projections": "object",
                        "competitive_risks": "array",
                        "investment_thesis": "string"
                    }
                }
            },
            {
                "agent": "tech_industry_trends_analyst",
                "content": "Research broader technology industry trends affecting Microsoft: cloud adoption, AI transformation, remote work, cybersecurity.",
                "tools": ["research:combined_search", "research:analyze_content"],
                "readFrom": ["microsoft_ai_strategy_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "cloud_adoption_trends": "object",
                        "ai_transformation_impact": "object",
                        "remote_work_influence": "object",
                        "cybersecurity_opportunities": "array"
                    }
                }
            },
            {
                "agent": "competitive_landscape_synthesizer",
                "content": "Synthesize all competitive intelligence into Microsoft's overall strategic position and recommendations.",
                "tools": ["research:generate_summary", "cite:format_citations"],
                "readFrom": ["google_competitive_threat_analyzer", "amazon_aws_threat_analyzer", "microsoft_ai_strategy_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "competitive_position": "string",
                        "key_strengths": "array",
                        "major_threats": "array",
                        "strategic_recommendations": "array"
                    }
                }
            },
            {
                "agent": "microsoft_investment_advisor",
                "content": "Create investment recommendation for Microsoft stock based on competitive analysis and market trends.",
                "tools": ["research:analyze_content", "cite:sources"],
                "readFrom": ["microsoft_financial_trend_analyzer", "competitive_landscape_synthesizer", "tech_industry_trends_analyst"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "investment_recommendation": "string",
                        "price_target": "string",
                        "key_catalysts": "array",
                        "risk_factors": "array",
                        "time_horizon": "string"
                    }
                }
            },
            {
                "agent": "executive_report_generator",
                "content": "Generate comprehensive executive report summarizing Microsoft's competitive position, opportunities, and strategic recommendations.",
                "tools": ["research:generate_summary", "cite:format_citations"],
                "readFrom": ["*"],
                "output_format": {
                    "type": "markdown",
                    "sections": [
                        "Executive Summary",
                        "Market Position Analysis", 
                        "Competitive Threats",
                        "Growth Opportunities",
                        "Financial Outlook",
                        "Strategic Recommendations"
                    ]
                }
            },
            {
                "agent": "workflow_analytics_monitor",
                "content": "Analyze workflow execution performance and data quality metrics.",
                "tools": ["research:analyze_content"],
                "output_format": {
                    "type": "json",
                    "schema": {
                        "execution_stats": "object",
                        "data_quality": "object",
                        "agent_performance": "array",
                        "recommendations": "array"
                    }
                }
            }
        ]
    }
    
    return workflow

# Main Execution Functions
async def execute_microsoft_workflow_fixed():
    """Execute Microsoft workflow with proper framework integration"""
    
    print("🚀 Microsoft Intelligence Workflow - Cross-Platform Version")
    print("=" * 60)
    
    workflow = create_simplified_microsoft_workflow()
    
    print(f"📊 Workflow Overview:")
    print(f"   Total Agents: {len(workflow['steps'])}")
    print(f"   Workflow: {workflow['workflow_name']}")
    print()
    
    # Save workflow to file
    workflow_file = "microsoft_simplified.json"
    with open(workflow_file, 'w', encoding='utf-8') as f:
        json.dump(workflow, f, indent=2)
    
    print(f"✅ Workflow saved to: {workflow_file}")
    
    try:
        # Discover tools
        print("🔧 Discovering available tools...")
        num_tools = tool_manager.discover_tools()
        print(f"   Discovered {num_tools} tools")
        
        # Show available research tools
        research_tools = [tool for tool in tool_manager.get_all_tools() if tool.startswith('research:')]
        print(f"   Available research tools: {len(research_tools)}")
        for tool in research_tools[:5]:
            print(f"     - {tool}")
        
        # Get configuration
        config = get_config()
        print(f"   API Endpoint: {config.get('endpoint', 'Not configured')}")
        print(f"   Model: {config.get('default_model', 'Not configured')}")
        
        # Check API key
        if not config.get('api_key'):
            print("⚠️  Warning: No API key configured. Some tools may not work.")
        
        print("\n🚀 Starting workflow execution...")
        start_time = time.time()
        
        # Execute workflow
        async with AsyncToolIntegratedExecutor(config, max_concurrent=8) as executor:
            results = await executor.execute_workflow_with_tools(workflow['steps'])
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Display results
        print(f"\n{'='*60}")
        print("🎉 WORKFLOW EXECUTION COMPLETED")
        print(f"{'='*60}")
        
        if isinstance(results, dict) and "results" in results:
            exec_results = results["results"]
            total_agents = len(workflow['steps'])
            completed_agents = results.get("completed_count", 0)
            failed_agents = results.get("failed_count", 0)
            success_rate = (completed_agents / total_agents) * 100 if total_agents > 0 else 0
            
            print(f"📈 Execution Statistics:")
            print(f"   Total Agents: {total_agents}")
            print(f"   Completed Successfully: {completed_agents}")
            print(f"   Failed: {failed_agents}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Execution Time: {execution_time:.1f} seconds")
            print(f"   Average Time per Agent: {execution_time/total_agents:.2f} seconds")
            
            # Show successful agents
            print(f"\n✅ Successful Agents:")
            for agent_name, result in exec_results.items():
                if not isinstance(result, dict) or "error" not in result:
                    print(f"   ✓ {agent_name}")
                    if isinstance(result, dict) and "content" in result:
                        content = str(result["content"])
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"     Preview: {preview}")
            
            # Show failed agents
            print(f"\n❌ Failed Agents:")
            failed_count = 0
            for agent_name, result in exec_results.items():
                if isinstance(result, dict) and "error" in result:
                    failed_count += 1
                    error_msg = result.get('error', 'Unknown error')
                    print(f"   ❌ {agent_name}")
                    print(f"     Error: {error_msg}")
            
            if failed_count == 0:
                print("   🎉 No failed agents!")
            
            # Save results
            timestamp = int(start_time)
            results_file = f"microsoft_results_{timestamp}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📁 Detailed results saved to: {results_file}")
            
            # Performance analysis
            if execution_time > 0:
                agents_per_second = completed_agents / execution_time
                print(f"\n🚀 Performance Metrics:")
                print(f"   Agents per Second: {agents_per_second:.2f}")
                print(f"   Total Tools Discovered: {num_tools}")
                print(f"   Framework: AsyncToolIntegratedExecutor")
            
            # Data quality check
            research_agents = [name for name in exec_results.keys() if 'research' in name or 'analyst' in name]
            print(f"\n📊 Data Quality:")
            print(f"   Research Agents: {len(research_agents)}")
            print(f"   Synthesis Agents: {len([n for n in exec_results.keys() if 'synthesizer' in n or 'generator' in n])}")
            
            return results
            
        else:
            print(f"❌ Unexpected result format: {type(results)}")
            return results
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all framework files are available")
        return None
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_individual_agent():
    """Test a single agent to verify tool integration"""
    
    print("\n🧪 Testing Individual Agent...")
    
    try:
        config = get_config()
        
        # Test single agent
        test_agent = {
            "agent": "test_microsoft_researcher",
            "content": "Research Microsoft's current stock price and recent news.",
            "tools": ["research:combined_search"],
            "output_format": {
                "type": "json",
                "schema": {
                    "stock_info": "object",
                    "recent_news": "array"
                }
            }
        }
        
        print(f"Testing agent: {test_agent['agent']}")
        
        async with AsyncToolIntegratedExecutor(config, max_concurrent=1) as executor:
            task = executor.create_task_from_step(test_agent, 0)
            result = await executor._execute_agent_with_tools(task)
        
        print(f"✅ Test agent result: {type(result)}")
        if isinstance(result, dict):
            if "error" in result:
                print(f"❌ Test failed: {result['error']}")
            else:
                print(f"✅ Test successful!")
                print(f"   Result keys: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ Individual agent test failed: {e}")
        return None

def run_workflow_from_file(workflow_path: str, data_file: str = None, max_concurrent: int = 5):
    """Run workflow from JSON file"""
    if not Path(workflow_path).exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        return None
    
    print(f"🚀 Loading workflow: {workflow_path}")
    
    try:
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow_data = json.load(f)
        
        if isinstance(workflow_data, dict) and "steps" in workflow_data:
            workflow = workflow_data["steps"]
        else:
            workflow = workflow_data
        
        print(f"📊 Loaded {len(workflow)} agents from {workflow_path}")
        
        async def execute_loaded_workflow():
            try:
                # Discover tools
                print("🔧 Discovering tools...")
                num_tools = tool_manager.discover_tools()
                print(f"   Discovered {num_tools} tools")
                
                # Get config
                config = get_config()
                
                # Execute workflow
                async with AsyncToolIntegratedExecutor(config, max_concurrent=max_concurrent) as executor:
                    results = await executor.execute_workflow_with_tools(workflow, data_file)
                
                return results
                
            except Exception as e:
                print(f"❌ Workflow execution failed: {e}")
                return None
        
        return asyncio.run(execute_loaded_workflow())
        
    except Exception as e:
        print(f"❌ Failed to load workflow: {e}")
        return None

def main():
    """Main execution function with CLI support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Platform Workflow Runner")
    parser.add_argument("--workflow", help="Workflow JSON file to execute")
    parser.add_argument("--data", help="Data file to process")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent agents")
    parser.add_argument("--microsoft", action="store_true", help="Run built-in Microsoft workflow")
    parser.add_argument("--test", action="store_true", help="Run test agent only")
    
    args = parser.parse_args()
    
    print("🏢 Cross-Platform Workflow Runner")
    print(f"Platform: {sys.platform}")
    print("=" * 60)
    
    if args.test:
        # Test individual agent
        test_result = asyncio.run(test_individual_agent())
        if test_result and isinstance(test_result, dict) and "error" not in test_result:
            print("✅ Test passed!")
        else:
            print("❌ Test failed!")
        return
    
    if args.workflow:
        # Run custom workflow
        print(f"Running custom workflow: {args.workflow}")
        results = run_workflow_from_file(args.workflow, args.data, args.concurrent)
        
        if results:
            print(f"\n✅ Workflow completed!")
            print(f"Completed: {results.get('completed_count', 0)}")
            print(f"Failed: {results.get('failed_count', 0)}")
        else:
            print("❌ Workflow failed!")
        
        return
    
    if args.microsoft:
        # Run built-in Microsoft workflow
        print("Running built-in Microsoft workflow...")
        results = asyncio.run(execute_microsoft_workflow_fixed())
        
        if results:
            print("\n🎯 Microsoft Workflow Analysis Complete!")
        else:
            print("❌ Microsoft workflow failed!")
        
        return
    
    # Default: Test agent first, then ask what to do
    print("No specific workflow specified. Running test...")
    test_result = asyncio.run(test_individual_agent())
    
    if test_result and isinstance(test_result, dict) and "error" not in test_result:
        print("\n✅ Test passed! System is working.")
        print("\nOptions:")
        print("1. Run Microsoft workflow: python script.py --microsoft")
        print("2. Run custom workflow: python script.py --workflow yourfile.json")
        print("3. Set concurrency: python script.py --workflow yourfile.json --concurrent 3")
    else:
        print("\n❌ Test failed! Check configuration:")
        print("1. Verify API key is set in config.py or environment")
        print("2. Check network connectivity")
        print("3. Ensure tool files are present")

if __name__ == "__main__":
    main()