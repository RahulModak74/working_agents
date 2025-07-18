#!/usr/bin/env python3

from sanic import Sanic, response
import json
import os
import uuid
import subprocess
from datetime import datetime
from pathlib import Path

app = Sanic("MinimalWorkflowAPI")

# Directories
UPLOAD_DIR = "./uploads"
WORKFLOWS_DIR = "./workflows" 
RESULTS_DIR = "./results"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, WORKFLOWS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Execution tracking
executions = {}

@app.route("/", methods=["GET"])
async def web_ui(request):
    """Serve minimal web interface"""
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Agentic Workflow Runner</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        .section { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
        button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background: #005a8b; }
        input[type="file"] { margin: 10px 0; }
        .results { background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .error { background: #ffebee; color: #c62828; }
        .success { background: #e8f5e8; color: #2e7d32; }
        pre { background: #f0f0f0; padding: 10px; overflow-x: auto; border-radius: 4px; }
    </style>
</head>
<body>
    <h1>🤖 Agentic Workflow Runner</h1>
    
    <!-- Workflow Upload -->
    <div class="section">
        <h3>1. Upload Workflow</h3>
        <input type="file" id="workflowFile" accept=".json" />
        <button onclick="uploadWorkflow()">Upload JSON Workflow</button>
        <div id="workflowResult"></div>
    </div>
    
    <!-- Data Upload (Optional) -->
    <div class="section">
        <h3>2. Upload Data Files (Optional)</h3>
        <input type="file" id="dataFiles" multiple accept=".csv,.xlsx,.json" />
        <button onclick="uploadData()">Upload Data Files</button>
        <div id="dataResult"></div>
    </div>
    
    <!-- Execution -->
    <div class="section">
        <h3>3. Execute Workflow</h3>
        <button onclick="executeWorkflow()">▶️ Run Workflow</button>
        <button onclick="executeWithData()">▶️ Run with Data</button>
        <div id="executionResult"></div>
    </div>
    
    <!-- Results -->
    <div class="section">
        <h3>4. Results</h3>
        <button onclick="checkStatus()">🔄 Check Status</button>
        <button onclick="downloadResults()">📥 Download Results</button>
        <div id="statusResult"></div>
    </div>

    <script>
        let currentWorkflowId = null;
        let currentDataFiles = [];
        let currentExecutionId = null;

        function showResult(elementId, message, isError = false) {
            const el = document.getElementById(elementId);
            el.innerHTML = `<div class="results ${isError ? 'error' : 'success'}">${message}</div>`;
        }

        async function uploadWorkflow() {
            const fileInput = document.getElementById('workflowFile');
            if (!fileInput.files[0]) {
                showResult('workflowResult', 'Please select a JSON workflow file', true);
                return;
            }

            const formData = new FormData();
            formData.append('workflow', fileInput.files[0]);

            try {
                const response = await fetch('/upload-workflow', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (response.ok) {
                    currentWorkflowId = result.workflow_id;
                    showResult('workflowResult', 
                        `✅ Workflow uploaded: ${result.steps} steps, ${result.agents.length} agents`);
                } else {
                    showResult('workflowResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('workflowResult', `❌ Upload failed: ${error.message}`, true);
            }
        }

        async function uploadData() {
            const fileInput = document.getElementById('dataFiles');
            if (!fileInput.files.length) {
                showResult('dataResult', 'Please select data files', true);
                return;
            }

            const formData = new FormData();
            Array.from(fileInput.files).forEach((file, index) => {
                formData.append(`data_${index}`, file);
            });

            try {
                const response = await fetch('/upload-data', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                
                if (response.ok) {
                    currentDataFiles = result.uploaded_files;
                    showResult('dataResult', 
                        `✅ Uploaded ${result.uploaded_files.length} data files`);
                } else {
                    showResult('dataResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('dataResult', `❌ Upload failed: ${error.message}`, true);
            }
        }

        async function executeWorkflow() {
            if (!currentWorkflowId) {
                showResult('executionResult', 'Please upload a workflow first', true);
                return;
            }

            try {
                const response = await fetch('/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        workflow_id: currentWorkflowId,
                        use_data: false
                    })
                });
                const result = await response.json();
                
                if (response.ok) {
                    currentExecutionId = result.execution_id;
                    showResult('executionResult', 
                        `✅ Execution started: ${result.execution_id}`);
                } else {
                    showResult('executionResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('executionResult', `❌ Execution failed: ${error.message}`, true);
            }
        }

        async function executeWithData() {
            if (!currentWorkflowId) {
                showResult('executionResult', 'Please upload a workflow first', true);
                return;
            }
            if (!currentDataFiles.length) {
                showResult('executionResult', 'Please upload data files first', true);
                return;
            }

            try {
                const response = await fetch('/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        workflow_id: currentWorkflowId,
                        data_files: currentDataFiles,
                        use_data: true
                    })
                });
                const result = await response.json();
                
                if (response.ok) {
                    currentExecutionId = result.execution_id;
                    showResult('executionResult', 
                        `✅ Execution with data started: ${result.execution_id}`);
                } else {
                    showResult('executionResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('executionResult', `❌ Execution failed: ${error.message}`, true);
            }
        }

        async function checkStatus() {
            if (!currentExecutionId) {
                showResult('statusResult', 'No execution in progress', true);
                return;
            }

            try {
                const response = await fetch(`/status/${currentExecutionId}`);
                const result = await response.json();
                
                if (response.ok) {
                    showResult('statusResult', 
                        `Status: ${result.status}<br>
                         Started: ${result.start_time}<br>
                         ${result.results_file ? '✅ Results ready' : '⏳ Running...'}`);
                } else {
                    showResult('statusResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('statusResult', `❌ Status check failed: ${error.message}`, true);
            }
        }

        async function downloadResults() {
            if (!currentExecutionId) {
                showResult('statusResult', 'No execution to download', true);
                return;
            }

            try {
                const response = await fetch(`/download/${currentExecutionId}`);
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `results_${currentExecutionId}.json`;
                    a.click();
                    showResult('statusResult', '✅ Results downloaded');
                } else {
                    const result = await response.json();
                    showResult('statusResult', `❌ ${result.error}`, true);
                }
            } catch (error) {
                showResult('statusResult', `❌ Download failed: ${error.message}`, true);
            }
        }
    </script>
</body>
</html>
    """
    return response.html(html)

@app.route("/upload-workflow", methods=["POST"])
async def upload_workflow(request):
    """Upload JSON workflow"""
    try:
        if 'workflow' not in request.files:
            return response.json({"error": "No workflow file"}, status=400)
        
        file = request.files['workflow'][0]
        workflow_id = str(uuid.uuid4())
        workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
        
        with open(workflow_path, 'wb') as f:
            f.write(file.body)
        
        # Analyze workflow
        with open(workflow_path, 'r') as f:
            workflow_data = json.load(f)
        
        steps = workflow_data if isinstance(workflow_data, list) else workflow_data.get("steps", [])
        agents = [step.get("agent", "unknown") for step in steps if step.get("agent")]
        
        return response.json({
            "workflow_id": workflow_id,
            "steps": len(steps),
            "agents": agents[:10],
            "filename": file.name
        })
        
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

@app.route("/upload-data", methods=["POST"])
async def upload_data(request):
    """Upload data files"""
    try:
        uploaded_files = []
        for field_name, file_list in request.files.items():
            for file_obj in file_list:
                file_id = str(uuid.uuid4())
                file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file_obj.name}")
                
                with open(file_path, 'wb') as f:
                    f.write(file_obj.body)
                
                uploaded_files.append({
                    "file_id": file_id,
                    "original_name": file_obj.name,
                    "file_path": file_path
                })
        
        return response.json({"uploaded_files": uploaded_files})
        
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

@app.route("/execute", methods=["POST"])
async def execute_workflow(request):
    """Execute workflow"""
    try:
        data = request.json
        workflow_id = data['workflow_id']
        use_data = data.get('use_data', False)
        data_files = data.get('data_files', [])
        
        workflow_path = os.path.join(WORKFLOWS_DIR, f"{workflow_id}.json")
        execution_id = str(uuid.uuid4())
        
        # Prepare command
        if use_data and data_files:
            # Use workflow_runner_v2.py with data
            cmd = ["python", "workflow_runner_v2.py", "--workflow", workflow_path]
            cmd.extend(["--data"] + [df["file_path"] for df in data_files[:3]])
        else:
            # Use workflow_runner_v1.py
            cmd = ["python", "workflow_runner_v1.py", workflow_path]
        
        # Execute workflow
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 cwd=".")
        
        executions[execution_id] = {
            "status": "running",
            "workflow_id": workflow_id,
            "start_time": datetime.now().isoformat(),
            "process": process,
            "use_data": use_data
        }
        
        return response.json({
            "execution_id": execution_id,
            "status": "started"
        })
        
    except Exception as e:
        return response.json({"error": str(e)}, status=500)

@app.route("/status/<execution_id>", methods=["GET"])
async def get_status(request, execution_id):
    """Get execution status"""
    if execution_id not in executions:
        return response.json({"error": "Execution not found"}, status=404)
    
    exec_info = executions[execution_id]
    process = exec_info.get("process")
    
    if process:
        if process.poll() is None:
            status = "running"
            results_file = None
        else:
            status = "completed" if process.returncode == 0 else "failed"
            # Look for results file
            workflow_name = Path(exec_info["workflow_id"]).stem
            results_file = f"{workflow_name}_results.json"
            if os.path.exists(results_file):
                exec_info["results_file"] = results_file
    else:
        status = exec_info["status"]
        results_file = exec_info.get("results_file")
    
    return response.json({
        "execution_id": execution_id,
        "status": status,
        "start_time": exec_info["start_time"],
        "results_file": results_file
    })

@app.route("/download/<execution_id>", methods=["GET"])
async def download_results(request, execution_id):
    """Download results file"""
    if execution_id not in executions:
        return response.json({"error": "Execution not found"}, status=404)
    
    exec_info = executions[execution_id]
    workflow_name = exec_info["workflow_id"]
    results_file = f"{workflow_name}_results.json"
    
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            file_content = f.read()
        return response.raw(file_content, 
                          headers={"Content-Disposition": f"attachment; filename={results_file}"},
                          content_type="application/json")
    else:
        return response.json({"error": "Results file not found"}, status=404)

if __name__ == "__main__":
    print("🚀 Minimal Workflow API Server")
    print("📋 Endpoints:")
    print("  GET  /           - Web Interface")
    print("  POST /upload-workflow - Upload JSON workflow")
    print("  POST /upload-data     - Upload data files")  
    print("  POST /execute         - Execute workflow")
    print("  GET  /status/<id>     - Check status")
    print("  GET  /download/<id>   - Download results")
    print("🌐 Access at: http://localhost:8000")
    
    app.run(host="0.0.0.0", port=8000, debug=True)