#!/usr/bin/env python3

from flask import Flask, request, jsonify, render_template, redirect, url_for
import os
import json
import subprocess
import threading
import time
import uuid
from datetime import datetime
import logging

# Import our workflow generator
try:
    from generate_research_workflow import parse_user_input, generate_workflow_file, run_workflow
except ImportError:
    print("Warning: Could not import workflow generator. Make sure generate_research_workflow.py is in the same directory.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("research_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("research_server")

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
WORKFLOW_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflows")
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WORKFLOW_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Store active research jobs
active_jobs = {}

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_research():
    """Handle research submission from the form."""
    try:
        # Extract data from the form
        research_topic = request.form.get('topic', '')
        research_description = request.form.get('description', '')
        dimensions = []
        
        # Get dimensions if provided
        for i in range(1, 4):  # Allow up to 3 dimensions
            dimension = request.form.get(f'dimension{i}', '').strip()
            if dimension:
                dimensions.append(dimension)
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create a CSV file with the input
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"research_input_{timestamp}.csv"
        input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
        
        with open(input_filepath, 'w', encoding='utf-8') as f:
            f.write(f"topic,description,dimension1,dimension2,dimension3\n")
            dims = dimensions + [''] * (3 - len(dimensions))  # Pad to 3 dimensions
            f.write(f"{research_topic},{research_description},{','.join(dims)}\n")
        
        # Start the research process in a background thread
        thread = threading.Thread(
            target=process_research_request,
            args=(job_id, input_filepath)
        )
        thread.daemon = True
        thread.start()
        
        # Store job information
        active_jobs[job_id] = {
            'topic': research_topic,
            'status': 'processing',
            'started_at': time.time(),
            'input_file': input_filepath,
            'workflow_file': None,
            'result_file': None
        }
        
        # Redirect to the status page
        return redirect(url_for('job_status', job_id=job_id))
        
    except Exception as e:
        logger.error(f"Error processing research submission: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to process research request: {str(e)}"
        }), 500

@app.route('/api/submit', methods=['POST'])
def api_submit_research():
    """API endpoint for submitting research requests."""
    try:
        # Check if we have JSON or form data
        if request.is_json:
            data = request.json
        else:
            data = request.form
        
        # Extract research parameters
        research_topic = data.get('topic', '')
        if not research_topic:
            return jsonify({
                'status': 'error',
                'message': 'Research topic is required'
            }), 400
        
        research_description = data.get('description', research_topic)
        
        # Extract dimensions
        dimensions = []
        if 'dimensions' in data and isinstance(data['dimensions'], list):
            dimensions = data['dimensions']
        else:
            # Check for individual dimension fields
            for i in range(1, 10):
                dimension_key = f'dimension{i}'
                if dimension_key in data and data[dimension_key]:
                    dimensions.append(data[dimension_key])
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Create a JSON file with the input
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"research_input_{timestamp}.json"
        input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
        
        with open(input_filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'topic': research_topic,
                'description': research_description,
                'dimensions': dimensions
            }, f, indent=2)
        
        # Start the research process in a background thread
        thread = threading.Thread(
            target=process_research_request,
            args=(job_id, input_filepath)
        )
        thread.daemon = True
        thread.start()
        
        # Store job information
        active_jobs[job_id] = {
            'topic': research_topic,
            'status': 'processing',
            'started_at': time.time(),
            'input_file': input_filepath,
            'workflow_file': None,
            'result_file': None
        }
        
        # Return the job ID
        return jsonify({
            'status': 'success',
            'message': 'Research request submitted successfully',
            'job_id': job_id
        })
        
    except Exception as e:
        logger.error(f"API error processing research submission: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f"Failed to process research request: {str(e)}"
        }), 500

@app.route('/status/<job_id>')
def job_status(job_id):
    """Show the status of a specific job."""
    if job_id not in active_jobs:
        return render_template('error.html', message=f"Job ID {job_id} not found"), 404
    
    job_info = active_jobs[job_id]
    
    # Calculate elapsed time
    elapsed_seconds = time.time() - job_info['started_at']
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    return render_template(
        'status.html',
        job_id=job_id,
        topic=job_info['topic'],
        status=job_info['status'],
        elapsed_time=elapsed_time,
        result_file=job_info.get('result_file')
    )

@app.route('/api/status/<job_id>')
def api_job_status(job_id):
    """API endpoint for job status."""
    if job_id not in active_jobs:
        return jsonify({
            'status': 'error',
            'message': f"Job ID {job_id} not found"
        }), 404
    
    job_info = active_jobs[job_id]
    
    # Calculate elapsed time
    elapsed_seconds = time.time() - job_info['started_at']
    
    response = {
        'job_id': job_id,
        'topic': job_info['topic'],
        'status': job_info['status'],
        'elapsed_seconds': elapsed_seconds,
        'started_at': job_info['started_at']
    }
    
    # Add result file if available
    if 'result_file' in job_info and job_info['result_file']:
        response['result_file'] = job_info['result_file']
    
    return jsonify(response)

@app.route('/jobs')
def list_jobs():
    """List all active and completed jobs."""
    return render_template('jobs.html', jobs=active_jobs)

@app.route('/api/jobs')
def api_list_jobs():
    """API endpoint for listing all jobs."""
    return jsonify({
        'jobs': active_jobs
    })

def process_research_request(job_id, input_filepath):
    """Process a research request in the background."""
    try:
        logger.info(f"Starting research process for job {job_id} with input {input_filepath}")
        
        # Update job status
        active_jobs[job_id]['status'] = 'generating_workflow'
        
        # Parse the input file
        research_request = parse_user_input(input_filepath)
        
        # Generate the workflow file
        workflow_path = generate_workflow_file(research_request, WORKFLOW_FOLDER)
        
        # Update job information
        active_jobs[job_id]['workflow_file'] = workflow_path
        active_jobs[job_id]['status'] = 'running_workflow'
        
        # Run the workflow
        run_workflow(workflow_path)
        
        # Find the output file (look for the most recent file in RESULTS_FOLDER)
        output_files = [os.path.join(RESULTS_FOLDER, f) for f in os.listdir(RESULTS_FOLDER)
                     if f.endswith('.md') or f.endswith('.json') or f.endswith('.txt')]
        
        if output_files:
            # Sort by modification time, newest first
            output_files.sort(key=os.path.getmtime, reverse=True)
            result_file = output_files[0]
            
            # Update job information
            active_jobs[job_id]['result_file'] = result_file
            active_jobs[job_id]['status'] = 'completed'
            logger.info(f"Research process for job {job_id} completed. Result: {result_file}")
        else:
            active_jobs[job_id]['status'] = 'completed_no_results'
            logger.warning(f"Research process for job {job_id} completed but no result files found")
        
    except Exception as e:
        logger.error(f"Error in research process for job {job_id}: {str(e)}")
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['error_message'] = str(e)

if __name__ == "__main__":
    # Create a basic templates directory and HTML files if they don't exist
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html if it doesn't exist
    index_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Advanced Research System</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        label { display: block; margin-top: 15px; font-weight: bold; }
        input[type="text"], textarea { width: 100%; padding: 8px; margin-top: 5px; box-sizing: border-box; }
        button { background-color: #4285f4; color: white; border: none; padding: 10px 20px; margin-top: 20px; cursor: pointer; }
        button:hover { background-color: #3367d6; }
        .dimension-container { margin-top: 10px; }
        .dimension-field { margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced Research System</h1>
        <p>Enter your research topic and optional details below.</p>
        
        <form action="/submit" method="post">
            <label for="topic">Research Topic:</label>
            <input type="text" id="topic" name="topic" required placeholder="Enter your research topic here">
            
            <label for="description">Description:</label>
            <textarea id="description" name="description" rows="4" placeholder="Provide more details about what you're looking to research"></textarea>
            
            <label>Optional Research Dimensions:</label>
            <div class="dimension-container">
                <div class="dimension-field">
                    <input type="text" name="dimension1" placeholder="Dimension 1 (e.g., Technical Capabilities)">
                </div>
                <div class="dimension-field">
                    <input type="text" name="dimension2" placeholder="Dimension 2 (e.g., Ethical Implications)">
                </div>
                <div class="dimension-field">
                    <input type="text" name="dimension3" placeholder="Dimension 3 (e.g., Economic Impact)">
                </div>
            </div>
            
            <button type="submit">Submit Research Request</button>
        </form>
        
        <p><a href="/jobs">View all research jobs</a></p>
    </div>
</body>
</html>""")
    
    # Create status.html if it doesn't exist
    status_path = os.path.join(templates_dir, "status.html")
    if not os.path.exists(status_path):
        with open(status_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Research Status</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
        }
        .status-processing { background-color: #2196F3; }
        .status-generating { background-color: #FF9800; }
        .status-running { background-color: #9C27B0; }
        .status-completed { background-color: #4CAF50; }
        .status-error { background-color: #F44336; }
        .info-row { margin: 15px 0; }
        .info-label { font-weight: bold; }
        .result-box { margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 4px; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            display: inline-block;
            margin-right: 10px;
            vertical-align: middle;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script>
        // Auto-refresh the page every 10 seconds if job is still processing
        function checkRefresh() {
            const status = document.getElementById('status-value').innerText.toLowerCase();
            if (status !== 'completed' && status !== 'error') {
                setTimeout(() => window.location.reload(), 10000);
            }
        }
        window.onload = checkRefresh;
    </script>
</head>
<body>
    <div class="container">
        <h1>Research Status</h1>
        
        <div class="info-row">
            <span class="info-label">Job ID:</span> {{ job_id }}
        </div>
        
        <div class="info-row">
            <span class="info-label">Topic:</span> {{ topic }}
        </div>
        
        <div class="info-row">
            <span class="info-label">Status:</span> 
            <span id="status-value" class="status-badge status-{{ status }}">
                {% if status == 'processing' or status == 'generating_workflow' or status == 'running_workflow' %}
                    <span class="spinner"></span>
                {% endif %}
                {{ status }}
            </span>
        </div>
        
        <div class="info-row">
            <span class="info-label">Elapsed Time:</span> {{ elapsed_time }}
        </div>
        
        {% if status == 'completed' and result_file %}
        <div class="result-box">
            <h3>Research Complete!</h3>
            <p>Your research results are ready. The system has generated a comprehensive report based on your query.</p>
            <p><a href="/results/{{ job_id }}" target="_blank">View Full Research Report</a></p>
        </div>
        {% elif status == 'error' %}
        <div class="result-box" style="background-color: #FFEBEE;">
            <h3>Error Processing Research</h3>
            <p>There was an error processing your research request. Please try again or contact support.</p>
        </div>
        {% elif status == 'processing' or status == 'generating_workflow' %}
        <div class="result-box" style="background-color: #E3F2FD;">
            <h3>Preparing Research Framework</h3>
            <p>The system is currently preparing the cognitive planning and optimization framework for your research query.</p>
            <p>This page will automatically refresh every 10 seconds.</p>
        </div>
        {% elif status == 'running_workflow' %}
        <div class="result-box" style="background-color: #E8EAF6;">
            <h3>Conducting Research</h3>
            <p>The advanced research system is actively gathering, analyzing, and synthesizing information about your topic.</p>
            <p>This process typically takes 5-20 minutes depending on the complexity of the topic.</p>
            <p>This page will automatically refresh every 10 seconds.</p>
        </div>
        {% endif %}
        
        <p><a href="/">Back to Home</a> | <a href="/jobs">View All Jobs</a></p>
    </div>
</body>
</html>""")
    
    # Create jobs.html if it doesn't exist
    jobs_path = os.path.join(templates_dir, "jobs.html")
    if not os.path.exists(jobs_path):
        with open(jobs_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Research Jobs</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        tr:hover { background-color: #f5f5f5; }
        .status-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            color: white;
            font-weight: bold;
            font-size: 0.8em;
        }
        .status-processing { background-color: #2196F3; }
        .status-generating { background-color: #FF9800; }
        .status-running { background-color: #9C27B0; }
        .status-completed { background-color: #4CAF50; }
        .status-error { background-color: #F44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Research Jobs</h1>
        
        {% if jobs %}
        <table>
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Topic</th>
                    <th>Status</th>
                    <th>Started</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {% for job_id, job in jobs.items() %}
                <tr>
                    <td>{{ job_id[:8] }}...</td>
                    <td>{{ job.topic }}</td>
                    <td>
                        <span class="status-badge status-{{ job.status }}">{{ job.status }}</span>
                    </td>
                    <td>{{ job.started_at|timestamp_to_date }}</td>
                    <td>
                        <a href="/status/{{ job_id }}">View Status</a>
                        {% if job.status == 'completed' and job.result_file %}
                        | <a href="/results/{{ job_id }}">View Results</a>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p>No research jobs found.</p>
        {% endif %}
        
        <p><a href="/">Submit New Research</a></p>
    </div>
</body>
</html>""")
    
    # Create error.html if it doesn't exist
    error_path = os.path.join(templates_dir, "error.html")
    if not os.path.exists(error_path):
        with open(error_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Error</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #F44336; }
        .error-box { background-color: #FFEBEE; padding: 15px; border-radius: 4px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Error</h1>
        
        <div class="error-box">
            <p>{{ message }}</p>
        </div>
        
        <p><a href="/">Back to Home</a></p>
    </div>
</body>
</html>""")
    
    # Register a filter to format timestamps
    @app.template_filter('timestamp_to_date')
    def timestamp_to_date(timestamp):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Add a route to view results
    @app.route('/results/<job_id>')
    def view_results(job_id):
        if job_id not in active_jobs:
            return render_template('error.html', message=f"Job ID {job_id} not found"), 404
        
        job_info = active_jobs[job_id]
        
        if job_info['status'] != 'completed' or not job_info.get('result_file'):
            return render_template('error.html', message=f"Results not yet available for job {job_id}"), 404
        
        result_file = job_info['result_file']
        
        # Read the content of the result file
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine the file type
            if result_file.endswith('.md'):
                # Render Markdown
                return render_template('markdown_viewer.html', content=content, job_id=job_id)
            elif result_file.endswith('.json'):
                # Pretty-print JSON
                json_content = json.loads(content)
                formatted_json = json.dumps(json_content, indent=2)
                return render_template('json_viewer.html', content=formatted_json, job_id=job_id)
            else:
                # Plain text
                return render_template('text_viewer.html', content=content, job_id=job_id)
        
        except Exception as e:
            logger.error(f"Error reading result file for job {job_id}: {str(e)}")
            return render_template('error.html', message=f"Error reading results: {str(e)}"), 500
    
    # Create Markdown viewer template
    markdown_viewer_path = os.path.join(templates_dir, "markdown_viewer.html")
    if not os.path.exists(markdown_viewer_path):
        with open(markdown_viewer_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Research Results</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.1.0/github-markdown.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.0.2/marked.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .header { background-color: #333; color: white; padding: 10px 20px; }
        .header a { color: white; text-decoration: none; margin-right: 20px; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        .markdown-body { padding: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <a href="/">Home</a>
        <a href="/status/{{ job_id }}">Status</a>
        <a href="/jobs">All Jobs</a>
    </div>
    
    <div class="container">
        <div id="content" class="markdown-body"></div>
    </div>
    
    <script>
        // Render the markdown content
        const markdownContent = `{{ content|safe }}`;
        document.getElementById('content').innerHTML = marked.parse(markdownContent);
    </script>
</body>
</html>""")
    
    # Create JSON viewer template
    json_viewer_path = os.path.join(templates_dir, "json_viewer.html")
    if not os.path.exists(json_viewer_path):
        with open(json_viewer_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Research Results (JSON)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .header { background-color: #333; color: white; padding: 10px 20px; }
        .header a { color: white; text-decoration: none; margin-right: 20px; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        pre { background-color: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <a href="/">Home</a>
        <a href="/status/{{ job_id }}">Status</a>
        <a href="/jobs">All Jobs</a>
    </div>
    
    <div class="container">
        <h1>Research Results (JSON)</h1>
        <pre><code>{{ content }}</code></pre>
    </div>
</body>
</html>""")
    
    # Create Text viewer template
    text_viewer_path = os.path.join(templates_dir, "text_viewer.html")
    if not os.path.exists(text_viewer_path):
        with open(text_viewer_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Research Results (Text)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
        .header { background-color: #333; color: white; padding: 10px 20px; }
        .header a { color: white; text-decoration: none; margin-right: 20px; }
        .container { max-width: 1000px; margin: 0 auto; padding: 20px; }
        pre { background-color: #f5f5f5; padding: 15px; border-radius: 4px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="header">
        <a href="/">Home</a>
        <a href="/status/{{ job_id }}">Status</a>
        <a href="/jobs">All Jobs</a>
    </div>
    
    <div class="container">
        <h1>Research Results</h1>
        <pre>{{ content }}</pre>
    </div>
</body>
</html>""")
    
    # Run the Flask application
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
