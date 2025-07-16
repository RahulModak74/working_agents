#!/usr/bin/env python3
import asyncio
import aiohttp
import json
import time
import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import importlib.util
import threading
import concurrent.futures
import signal
import atexit

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("async_platform")

@dataclass
class AgentTask:
    agent_name: str
    prompt: str
    file_path: Optional[str] = None
    output_format: Optional[Dict] = None
    references: Optional[Dict] = None
    tools: List[str] = None
    dependencies: Set[str] = None
    step_index: int = 0

class WindowsAsyncExecutor:
    def __init__(self, config: Dict, max_concurrent: int = 10):
        self.config = config
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        self.connector = None
        self.results = {}
        self.request_times = deque()
        self.rate_limit = 2.0
        
    async def __aenter__(self):
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
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()
    
    async def _rate_limit(self):
        async with self.semaphore:
            now = time.time()
            self.request_times = deque([t for t in self.request_times if now - t < 1.0])
            if len(self.request_times) >= self.rate_limit:
                await asyncio.sleep(1.0 - (now - self.request_times[0]))
            self.request_times.append(now)
    
    async def _call_api(self, messages: List[Dict], retries: int = 3) -> str:
        await self._rate_limit()
        
        payload = {
            "model": self.config["default_model"],
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        headers = {}
        if self.config.get("api_key"):
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        
        for attempt in range(retries):
            try:
                async with self.session.post(self.config["endpoint"], json=payload, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('content', '') or data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    elif response.status == 429:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        break
            except Exception as e:
                if attempt == retries - 1:
                    return f"Error: {str(e)}"
                await asyncio.sleep(2 ** attempt)
        
        return "Error: API request failed"
    
    async def _execute_agent(self, task: AgentTask) -> Dict[str, Any]:
        try:
            prompt = task.prompt
            
            if task.references:
                prompt += "\n\nReference Information:\n"
                for ref_name, ref_content in task.references.items():
                    prompt += f"\n{ref_name}:\n{json.dumps(ref_content) if isinstance(ref_content, dict) else str(ref_content)}\n"
            
            if task.file_path and Path(task.file_path).exists():
                content = Path(task.file_path).read_text(encoding='utf-8')[:5000]
                prompt += f"\n\nFile content:\n{content}"
            
            if task.output_format and task.output_format.get("type") == "json":
                prompt += f"\n\nRespond with valid JSON matching: {json.dumps(task.output_format.get('schema', {}))}"
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Follow instructions precisely."},
                {"role": "user", "content": prompt}
            ]
            
            response = await self._call_api(messages)
            
            if task.output_format and task.output_format.get("type") == "json":
                try:
                    return json.loads(response)
                except json.JSONDecodeError:
                    import re
                    match = re.search(r'\{.*\}', response, re.DOTALL)
                    if match:
                        try:
                            return json.loads(match.group())
                        except:
                            pass
                    return {"error": "Invalid JSON", "raw_response": response}
            else:
                return {"content": response}
                
        except Exception as e:
            return {"error": str(e)}
    
    def _build_dependencies(self, workflow: List[Dict]) -> Dict[str, Set[str]]:
        deps = {}
        for step in workflow:
            agent_name = step.get("agent", "")
            if not agent_name:
                continue
            
            agent_deps = set()
            for ref in step.get("readFrom", []):
                if isinstance(ref, str) and ref != "*":
                    agent_deps.add(ref)
                elif ref == "*":
                    for prev_step in workflow:
                        if prev_step.get("agent") == agent_name:
                            break
                        prev_agent = prev_step.get("agent")
                        if prev_agent:
                            agent_deps.add(prev_agent)
            
            deps[agent_name] = agent_deps
        
        return deps
    
    def _create_batches(self, workflow: List[Dict]) -> List[List[AgentTask]]:
        deps = self._build_dependencies(workflow)
        
        tasks = {}
        for i, step in enumerate(workflow):
            agent_name = step.get("agent", "")
            if agent_name:
                tasks[agent_name] = AgentTask(
                    agent_name=agent_name,
                    prompt=step.get("content", ""),
                    file_path=step.get("file"),
                    output_format=step.get("output_format"),
                    tools=step.get("tools", []),
                    dependencies=deps.get(agent_name, set()),
                    step_index=i
                )
        
        batches = []
        completed = set()
        remaining = set(tasks.keys())
        
        while remaining:
            ready = []
            for agent_name in remaining:
                if tasks[agent_name].dependencies.issubset(completed):
                    ready.append(tasks[agent_name])
            
            if not ready:
                ready = [tasks[name] for name in list(remaining)[:self.max_concurrent]]
            
            batch = ready[:self.max_concurrent]
            batches.append(batch)
            
            for task in batch:
                completed.add(task.agent_name)
                remaining.discard(task.agent_name)
        
        return batches
    
    async def execute_workflow(self, workflow: List[Dict], data_file: str = None) -> Dict[str, Any]:
        batches = self._create_batches(workflow)
        
        for batch_idx, batch in enumerate(batches):
            logger.info(f"Executing batch {batch_idx + 1}/{len(batches)}: {[t.agent_name for t in batch]}")
            
            for task in batch:
                refs = {}
                for dep in task.dependencies:
                    if dep in self.results:
                        refs[dep] = self.results[dep]
                task.references = refs
            
            batch_results = await asyncio.gather(*[self._execute_agent(task) for task in batch], return_exceptions=True)
            
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.results[task.agent_name] = {"error": str(result)}
                else:
                    self.results[task.agent_name] = result
        
        return {
            "results": self.results,
            "completed_count": len([r for r in self.results.values() if "error" not in r]),
            "failed_count": len([r for r in self.results.values() if "error" in r]),
            "total_count": len(workflow)
        }

class ToolManager:
    def __init__(self):
        self.tools = {}
        self.modules = set()
    
    def discover_tools(self, directories: List[str] = None) -> int:
        if directories is None:
            directories = [Path.cwd()]
        
        count = 0
        for directory in directories:
            for py_file in Path(directory).glob("*.py"):
                if py_file.stem in ['__init__', 'config', 'main'] or py_file.stem in self.modules:
                    continue
                
                try:
                    spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        if hasattr(module, 'TOOL_REGISTRY'):
                            for tool_id, handler in module.TOOL_REGISTRY.items():
                                self.tools[tool_id] = handler
                                count += 1
                        
                        self.modules.add(py_file.stem)
                except Exception as e:
                    logger.debug(f"Failed to load {py_file}: {e}")
        
        return count
    
    def execute_tool(self, tool_id: str, **kwargs) -> Any:
        if tool_id not in self.tools:
            return {"error": f"Tool not found: {tool_id}"}
        
        try:
            return self.tools[tool_id](**kwargs)
        except Exception as e:
            return {"error": str(e)}
    
    def get_all_tools(self) -> List[str]:
        return list(self.tools.keys())

class ConfigManager:
    @staticmethod
    def get_config() -> Dict[str, Any]:
        config_paths = [
            Path.cwd() / "config.py",
            Path.cwd() / "openrouter_config.py",
            Path.cwd() / "ollama_config.py"
        ]
        
        for config_path in config_paths:
            if config_path.exists():
                try:
                    spec = importlib.util.spec_from_file_location("config", config_path)
                    config_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(config_module)
                    return config_module.CONFIG
                except Exception as e:
                    logger.debug(f"Failed to load {config_path}: {e}")
        
        return {
            "default_model": "deepseek/deepseek-chat:free",
            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "output_dir": str(Path.cwd() / "outputs")
        }

class WorkflowRunner:
    def __init__(self, config: Dict = None, max_concurrent: int = 5):
        self.config = config or ConfigManager.get_config()
        self.max_concurrent = max_concurrent
        self.tool_manager = ToolManager()
        
    def load_workflow(self, workflow_path: str) -> List[Dict]:
        workflow_file = Path(workflow_path)
        if not workflow_file.exists():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")
        
        data = json.loads(workflow_file.read_text(encoding='utf-8'))
        return data.get("steps", data) if isinstance(data, dict) else data
    
    async def run_async(self, workflow_path: str, data_file: str = None) -> Dict[str, Any]:
        workflow = self.load_workflow(workflow_path)
        
        async with WindowsAsyncExecutor(self.config, self.max_concurrent) as executor:
            results = await executor.execute_workflow(workflow, data_file)
        
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        results_file = output_dir / f"results_{int(time.time())}.json"
        results_file.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
        
        return results
    
    def run(self, workflow_path: str, data_file: str = None) -> Dict[str, Any]:
        return asyncio.run(self.run_async(workflow_path, data_file))

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Windows-Compatible Async Workflow Platform")
    parser.add_argument("--workflow", required=True, help="Workflow JSON file")
    parser.add_argument("--data", help="Data file to process")
    parser.add_argument("--concurrent", type=int, default=5, help="Max concurrent agents")
    parser.add_argument("--config", help="Config file path")
    
    args = parser.parse_args()
    
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            config = config_module.CONFIG
    
    runner = WorkflowRunner(config, args.concurrent)
    
    try:
        print(f"Starting workflow: {args.workflow}")
        print(f"Max concurrent: {args.concurrent}")
        
        start_time = time.time()
        results = runner.run(args.workflow, args.data)
        end_time = time.time()
        
        print(f"\nExecution completed in {end_time - start_time:.1f}s")
        print(f"Completed: {results['completed_count']}/{results['total_count']}")
        print(f"Failed: {results['failed_count']}")
        
        if results['failed_count'] > 0:
            print("\nFailed agents:")
            for agent, result in results['results'].items():
                if isinstance(result, dict) and 'error' in result:
                    print(f"  {agent}: {result['error']}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()