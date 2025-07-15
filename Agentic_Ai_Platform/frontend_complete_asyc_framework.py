#!/usr/bin/env python3

from sanic import Sanic, response
from sanic.request import Request
import json
import os
import asyncio
import uuid
from datetime import datetime
import traceback
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import time
from collections import defaultdict

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 10  # requests per minute
RATE_LIMIT_WINDOW = 60    # seconds
request_history = defaultdict(list)

def check_rate_limit(identifier="default"):
    """Check if request is within rate limit"""
    now = time.time()
    
    # Clean old requests
    request_history[identifier] = [
        req_time for req_time in request_history[identifier] 
        if now - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if under limit
    if len(request_history[identifier]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    request_history[identifier].append(now)
    return True

# Import your complete async framework
try:
    from complete_async_framework import CompleteAsyncFrameworkManager, run_complete_async_workflow
    COMPLETE_FRAMEWORK_AVAILABLE = True
    print("✅ Complete Async Framework loaded successfully")
except ImportError as e:
    print(f"⚠️ Complete async framework not found: {e}")
    COMPLETE_FRAMEWORK_AVAILABLE = False

try:
    from tool_manager import tool_manager
    print("✅ Tool manager loaded")
except ImportError as e:
    print(f"⚠️ Tool manager not found: {e}")
    tool_manager = None

try:
    from utils import get_config
    print("✅ Utils loaded")
except ImportError as e:
    print(f"⚠️ Utils not found: {e}")
    def get_config():
        return {
            "output_dir": "./agent_outputs",
            "default_model": "qwen2.5:7b",
            "api_key": "",
            "endpoint": "http://localhost:11434/v1/chat/completions",
            "timeout": 1200
        }

# Fallback imports
try:
    from async_framework_main import run_async_workflow
    ASYNC_FRAMEWORK_AVAILABLE = True
except ImportError:
    ASYNC_FRAMEWORK_AVAILABLE = False

try:
    from workflow_fix import run_workflow_with_real_calls
    WORKFLOW_FIX_AVAILABLE = True
except ImportError:
    WORKFLOW_FIX_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("workflow_executor")

app = Sanic("WorkflowExecutor")

# Configuration
UPLOAD_DIR = "./uploads"
WORKFLOWS_DIR = "./workflows"
RESULTS_DIR = "./results"
TEMPLATES_DIR = "./templates"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, WORKFLOWS_DIR, RESULTS_DIR, TEMPLATES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# In-memory execution tracking
executions = {}

# Get configuration
try:
    config = get_config()
    logger.info(f"✅ Configuration loaded: {config.get('endpoint', 'unknown')}")
except Exception as e:
    logger.error(f"❌ Failed to load configuration: {e}")
    config = {
        "output_dir": "./agent_outputs",
        "default_model": "qwen2.5:7b",
        "api_key": "",
        "endpoint": "http://localhost:11434/v1/chat/completions",
        "timeout": 1200
    }

# Update config with slower rate limits
config.update({
    "rate_limit_per_second": 0.5,  # Very slow - 1 request per 2 seconds
    "timeout": 300,  # 5 minutes timeout
    "max_retries": 2,  # Reduce retries
    "retry_delay": 10  # 10 second delay between retries
})

logger.info(f"🐌 Rate limiting enabled: {config.get('rate_limit_per_second', 1)} requests/second")

def analyze_workflow_features(steps: list) -> list:
    """Analyze workflow to determine required features"""
    if not steps:
        return []
    
    features = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("type") == "dynamic":
            features.append("dynamic_agents")
        if step.get("type") in ["loop", "conditional", "parallel", "state"]:
            features.append("flow_control")
        if step.get("tools"):
            features.append("tools")
    
    return list(set(features))

# API Routes
@app.route("/health", methods=["GET"])
async def health_check(request: Request):
    """Enhanced health check endpoint"""
    
    # Check framework availability
    frameworks_available = {
        "complete_async_framework": COMPLETE_FRAMEWORK_AVAILABLE,
        "async_framework_main": ASYNC_FRAMEWORK_AVAILABLE,
        "workflow_fix": WORKFLOW_FIX_AVAILABLE
    }
    
    # Check tool manager
    tool_count = 0
    if tool_manager:
        try:
            tool_count = tool_manager.discover_tools()
        except Exception as e:
            logger.warning(f"Tool discovery failed: {e}")
    
    # Test API endpoint if possible
    api_status = "unknown"
    try:
        import requests
        headers = {"Content-Type": "application/json"}
        if config.get("api_key"):
            headers["Authorization"] = f"Bearer {config['api_key']}"
        
        test_payload = {
            "model": config["default_model"],
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 1
        }
        
        response_test = requests.post(config["endpoint"], json=test_payload, headers=headers, timeout=5)
        api_status = "accessible" if response_test.status_code in [200, 401, 403] else f"error_{response_test.status_code}"
    except Exception as e:
        api_status = f"error: {str(e)[:50]}"
    
    return response.json({
        "status": "healthy",
        "service": "Complete Async Workflow Executor",
        "timestamp": datetime.now().isoformat(),
        "framework_status": frameworks_available,
        "config": {
            "model": config["default_model"],
            "endpoint": config["endpoint"],
            "api_status": api_status
        },
        "tools_discovered": tool_count,
        "directories": {
            "uploads": len(os.listdir(UPLOAD_DIR)),
            "workflows": len(os.listdir(WORKFLOWS_DIR)),
            "results": len(os.listdir(RESULTS_DIR))
        }
    })

@app.route("/test", methods=["GET"])
async def test_route(request: Request):
    """Simple test route"""
    return response.json({
        "status": "working", 
        "message": "Server is running",
        "frameworks": {
            "complete_framework": COMPLETE_FRAMEWORK_AVAILABLE,
            "async_framework": ASYNC_FRAMEWORK_AVAILABLE,
            "workflow_fix": WORKFLOW_FIX_AVAILABLE
        }
    })

@app.route("/upload-workflow", methods=["POST"])
async def upload_workflow(request: Request):
    """Upload a JSON workflow file"""
    try:
        logger.info("📤 Upload workflow request received")
        
        if not request.files:
            logger.error("No files in request")
            return response.json({"error": "No files in request"}, status=400)
        
        logger.info(f"Files received: {list(request.files.keys())}")
        
        if 'workflow' not in request.files:
            logger.error("No workflow file provided")
            return response.json({"error": "No workflow file provided"}, status=400)
        
        workflow_file = request.files['workflow'][0]
        workflow_id = str(uuid.uuid4())
        
        logger.info(f"Processing workflow: {workflow_file.name}, Size: {len(workflow_file.body)} bytes")
        
        # Save workflow file
        workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
        with open(workflow_path, 'wb') as f:
            f.write(workflow_file.body)
        
        # Validate and analyze JSON
        try:
            with open(workflow_path, 'r', encoding='utf-8') as f:
                workflow_data = json.load(f)
        except json.JSONDecodeError as e:
            os.remove(workflow_path)
            logger.error(f"Invalid JSON: {e}")
            return response.json({"error": f"Invalid JSON: {str(e)}"}, status=400)
        
        # Extract workflow steps
        if isinstance(workflow_data, list):
            steps = workflow_data
        elif isinstance(workflow_data, dict):
            steps = workflow_data.get("steps", workflow_data.get("workflow", []))
        else:
            steps = []
        
        # Analyze workflow features
        features = analyze_workflow_features(steps)
        
        # Get step count and agents
        step_count = len(steps)
        agent_names = [step.get("agent", "unknown") for step in steps if step.get("agent")]
        
        logger.info(f"✅ Workflow analyzed: {step_count} steps, {len(agent_names)} agents, features: {features}")
        
        return response.json({
            "workflow_id": workflow_id,
            "workflow_path": workflow_path,
            "status": "uploaded",
            "steps": step_count,
            "agents": agent_names[:10],  # First 10 agents
            "features": features,
            "filename": workflow_file.name,
            "message": "Workflow uploaded and analyzed successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}")
        traceback.print_exc()
        return response.json({"error": str(e)}, status=500)

@app.route("/upload-data", methods=["POST"])
async def upload_data(request: Request):
    """Upload data files for workflow execution"""
    try:
        if not request.files:
            return response.json({"error": "No files provided"}, status=400)
        
        uploaded_files = []
        
        for field_name, file_list in request.files.items():
            for file_obj in file_list:
                file_id = str(uuid.uuid4())
                safe_filename = f"{file_id}_{file_obj.name}"
                file_path = os.path.join(UPLOAD_DIR, safe_filename)
                
                with open(file_path, 'wb') as f:
                    f.write(file_obj.body)
                
                uploaded_files.append({
                    "field_name": field_name,
                    "original_name": file_obj.name,
                    "file_id": file_id,
                    "file_path": file_path,
                    "size": len(file_obj.body)
                })
                
                logger.info(f"📂 Data file uploaded: {file_obj.name} -> {file_path}")
        
        return response.json({
            "uploaded_files": uploaded_files,
            "message": f"Uploaded {len(uploaded_files)} files successfully"
        })
        
    except Exception as e:
        logger.error(f"❌ Data upload error: {str(e)}")
        return response.json({"error": str(e)}, status=500)

@app.route("/execute-workflow", methods=["POST"])
async def execute_workflow(request: Request):
    """Execute a workflow using the complete async framework"""
    try:
        data = request.json
        workflow_id = data.get('workflow_id')
        data_file_path = data.get('data_file_path')
        max_concurrent = data.get('max_concurrent', 3)  # Default to 3
        features = data.get('features', None)
        verbose = data.get('verbose', False)
        
        if not workflow_id:
            return response.json({"error": "workflow_id required"}, status=400)
        
        workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
        
        if not os.path.exists(workflow_path):
            return response.json({"error": f"Workflow not found: {workflow_id}"}, status=404)
        
        # Validate data file if provided
        if data_file_path and not os.path.exists(data_file_path):
            return response.json({"error": f"Data file not found: {data_file_path}"}, status=404)
        
        # Create execution record
        execution_id = str(uuid.uuid4())
        executions[execution_id] = {
            "status": "started",
            "workflow_id": workflow_id,
            "workflow_path": workflow_path,
            "start_time": datetime.now().isoformat(),
            "data_file": data_file_path,
            "max_concurrent": max_concurrent,
            "features": features,
            "verbose": verbose
        }
        
        logger.info(f"🚀 Starting execution {execution_id} with {max_concurrent} concurrent agents")
        
        # Start execution in background
        asyncio.create_task(run_complete_workflow_async(
            execution_id, workflow_path, data_file_path, max_concurrent, features, verbose
        ))
        
        return response.json({
            "execution_id": execution_id,
            "status": "started",
            "workflow_path": workflow_path,
            "data_file": data_file_path,
            "max_concurrent": max_concurrent,
            "framework": "complete_async_framework" if COMPLETE_FRAMEWORK_AVAILABLE else "fallback",
            "message": "Workflow execution started with complete async framework"
        })
        
    except Exception as e:
        logger.error(f"❌ Execute error: {str(e)}")
        return response.json({"error": str(e)}, status=500)

@app.route("/execution-status/<execution_id>", methods=["GET"])
async def get_execution_status(request: Request, execution_id: str):
    """Get execution status and results"""
    if execution_id not in executions:
        return response.json({"error": "Execution not found"}, status=404)
    
    execution_info = executions[execution_id].copy()
    
    # Check if results file exists
    results_file = os.path.join(RESULTS_DIR, f"{execution_id}_results.json")
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Extract summary information
            if isinstance(results, dict):
                if "results" in results:
                    exec_results = results["results"]
                    execution_info["results_summary"] = {
                        "total_agents": results.get("total_count", len(exec_results)),
                        "completed": results.get("completed_count", len(exec_results)),
                        "failed": results.get("failed_count", 0),
                        "execution_time": results.get("execution_time", 0),
                        "agents_per_second": results.get("completed_count", 0) / max(results.get("execution_time", 1), 1)
                    }
                else:
                    execution_info["results_summary"] = {
                        "total_agents": len([k for k in results.keys() if k != 'workflow_state']),
                        "completed": len([k for k in results.keys() if k != 'workflow_state']),
                        "failed": 0,
                        "execution_time": 0
                    }
                
                execution_info["results"] = results
                
        except Exception as e:
            execution_info["results_error"] = str(e)
    
    return response.json(execution_info)

@app.route("/list-workflows", methods=["GET"])
async def list_workflows(request: Request):
    """List all uploaded workflows with analysis"""
    try:
        workflows = []
        for filename in os.listdir(WORKFLOWS_DIR):
            if filename.endswith('.json'):
                workflow_id = filename[:-5]
                workflow_path = os.path.join(WORKFLOWS_DIR, filename)
                
                try:
                    with open(workflow_path, 'r', encoding='utf-8') as f:
                        workflow_data = json.load(f)
                    
                    if isinstance(workflow_data, list):
                        steps = workflow_data
                    elif isinstance(workflow_data, dict):
                        steps = workflow_data.get("steps", workflow_data.get("workflow", []))
                    else:
                        steps = []
                    
                    features = analyze_workflow_features(steps)
                    agent_names = [step.get("agent", "unknown") for step in steps if step.get("agent")]
                    
                    workflows.append({
                        "workflow_id": workflow_id,
                        "steps": len(steps),
                        "agents": agent_names[:5],
                        "features": features,
                        "created": datetime.fromtimestamp(os.path.getctime(workflow_path)).isoformat(),
                        "size_kb": round(os.path.getsize(workflow_path) / 1024, 1)
                    })
                    
                except Exception as e:
                    workflows.append({
                        "workflow_id": workflow_id,
                        "error": str(e)
                    })
        
        return response.json({"workflows": workflows})
        
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

@app.route("/list-executions", methods=["GET"])
async def list_executions(request: Request):
    """List all workflow executions"""
    try:
        enhanced_executions = {}
        for exec_id, exec_info in executions.items():
            enhanced_info = exec_info.copy()
            
            results_file = os.path.join(RESULTS_DIR, f"{exec_id}_results.json")
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    if isinstance(results, dict) and "results" in results:
                        enhanced_info["completion_summary"] = {
                            "completed": results.get("completed_count", 0),
                            "failed": results.get("failed_count", 0),
                            "total": results.get("total_count", 0)
                        }
                except:
                    pass
            
            enhanced_executions[exec_id] = enhanced_info
        
        return response.json({"executions": enhanced_executions})
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

@app.route("/framework-info", methods=["GET"])
async def framework_info(request: Request):
    """Get information about available frameworks and tools"""
    try:
        frameworks = {
            "complete_async_framework": {
                "available": COMPLETE_FRAMEWORK_AVAILABLE,
                "description": "Complete async framework with all features"
            },
            "async_framework_main": {
                "available": ASYNC_FRAMEWORK_AVAILABLE,
                "description": "Main async framework"
            },
            "workflow_fix": {
                "available": WORKFLOW_FIX_AVAILABLE,
                "description": "Workflow fix for real API calls"
            }
        }
        
        tools_info = {"available": False, "count": 0, "tools": []}
        if tool_manager:
            try:
                tool_count = tool_manager.discover_tools()
                tools_info = {
                    "available": True,
                    "count": tool_count,
                    "tools": tool_manager.get_all_tools()[:20],
                    "stats": tool_manager.get_stats()
                }
            except Exception as e:
                tools_info["error"] = str(e)
        
        return response.json({
            "frameworks": frameworks,
            "tools": tools_info,
            "config": {
                "model": config["default_model"],
                "endpoint": config["endpoint"],
                "output_dir": config.get("output_dir", "./agent_outputs")
            }
        })
        
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

async def run_complete_workflow_async(execution_id: str, workflow_path: str, 
                                    data_file_path: str = None, max_concurrent: int = 3,
                                    features: list = None, verbose: bool = False):
    """Execute workflow asynchronously using complete framework"""
    try:
        logger.info(f"🚀 [{execution_id}] Starting complete async workflow execution")
        executions[execution_id]["status"] = "running"
        
        # Rate limit check
        if not check_rate_limit(execution_id):
            logger.warning(f"⚠️ [{execution_id}] Rate limit exceeded, waiting...")
            await asyncio.sleep(30)  # Wait 30 seconds
        
        # Choose the best available framework
        if COMPLETE_FRAMEWORK_AVAILABLE:
            logger.info(f"🔧 [{execution_id}] Using CompleteAsyncFrameworkManager")
            
            # Create framework manager with slower settings
            framework_manager = CompleteAsyncFrameworkManager(config, verbose)
            results = await framework_manager.execute_workflow(
                workflow_file=workflow_path,
                data_file=data_file_path,
                max_concurrent=min(max_concurrent, 3),  # Limit to 3
                features=features
            )
            
        elif ASYNC_FRAMEWORK_AVAILABLE:
            logger.info(f"🔧 [{execution_id}] Using async_framework_main")
            results = await run_async_workflow(
                workflow_file=workflow_path,
                config=config,
                data_file=data_file_path,
                max_concurrent=min(max_concurrent, 3),  # Limit to 3
                features=features
            )
        elif WORKFLOW_FIX_AVAILABLE:
            logger.info(f"🔧 [{execution_id}] Using workflow_fix")
            results = run_workflow_with_real_calls(
                workflow_file=workflow_path,
                data_file=data_file_path,
                max_concurrent=min(max_concurrent, 3)  # Limit to 3
            )
        else:
            raise Exception("No suitable framework available")
        
        # Save results
        results_file = os.path.join(RESULTS_DIR, f"{execution_id}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Update execution status
        execution_time = (datetime.now() - datetime.fromisoformat(executions[execution_id]["start_time"])).total_seconds()
        
        executions[execution_id].update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "results_file": results_file,
            "execution_time": execution_time
        })
        
        # Extract summary
        if isinstance(results, dict):
            if "results" in results:
                completed = results.get("completed_count", 0)
                failed = results.get("failed_count", 0)
                total = results.get("total_count", 0)
            else:
                completed = len([k for k in results.keys() if k != 'workflow_state'])
                failed = len([k for k, v in results.items() if isinstance(v, dict) and v.get('error')])
                total = completed + failed
            
            executions[execution_id]["summary"] = {
                "completed": completed,
                "failed": failed,
                "total": total,
                "success_rate": (completed / total * 100) if total > 0 else 0
            }
        
        logger.info(f"✅ [{execution_id}] Workflow execution completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        logger.error(f"❌ [{execution_id}] Workflow execution failed: {error_msg}")
        
        executions[execution_id].update({
            "status": "failed",
            "end_time": datetime.now().isoformat(),
            "error": error_msg,
            "traceback": error_trace
        })

# Main web interface route
@app.route("/", methods=["GET"])
async def web_ui(request: Request):
    """Serve the main web interface"""
    template_path = os.path.join(TEMPLATES_DIR, "index.html")
    
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return response.html(html_content)
    else:
        return response.json({
            "error": "Template not found", 
            "message": "Please create templates/index.html file",
            "template_path": template_path
        }, status=404)

if __name__ == "__main__":
    print("🚀 Starting Complete Async Workflow Executor API")
    print("📋 Available endpoints:")
    print("  GET  /health              - Enhanced health check")
    print("  GET  /test                - Simple test endpoint")
    print("  GET  /                    - Main web interface")
    print("  POST /upload-workflow     - Upload and analyze JSON workflow")
    print("  POST /upload-data         - Upload data files")
    print("  POST /execute-workflow    - Execute workflow with complete framework")
    print("  GET  /execution-status/<id> - Check execution status")
    print("  GET  /list-workflows      - List all workflows with analysis")
    print("  GET  /list-executions     - List all executions")
    print("  GET  /framework-info      - Get framework and tools information")
    print("🌐 Access web dashboard at: http://localhost:8000")
    print("📋 API health check: http://localhost:8000/health")
    print("🧪 Test endpoint: http://localhost:8000/test")
    
    # Create directories if they don't exist
    for dir_path in [UPLOAD_DIR, WORKFLOWS_DIR, RESULTS_DIR, TEMPLATES_DIR]:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Directory ready: {dir_path}")
    
    # Log framework availability
    if COMPLETE_FRAMEWORK_AVAILABLE:
        print("✅ Complete Async Framework available")
    else:
        print("⚠️  Complete Async Framework not available, using fallback")
    
    # Log configuration
    print(f"🤖 Model: {config['default_model']}")
    print(f"🌐 Endpoint: {config['endpoint']}")
    print(f"🐌 Rate limit: {config['rate_limit_per_second']} requests/second")
    
    app.run(host="0.0.0.0", port=8000, debug=True)