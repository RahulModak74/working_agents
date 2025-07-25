#!/usr/bin/env python3
"""
Production Sanic API Server for Security Analysis
Simple two-page setup: Login + Analysis Dashboard
"""

from sanic import Sanic, Request, response
from sanic.response import json as sanic_json, html, file, redirect
from sanic.exceptions import InvalidUsage, ServerError
from sanic_cors import CORS
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import traceback

# Import our production analyzer
from production_pyro_analyzer import ProductionSecurityAnalyzer, create_analyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Sanic app
app = Sanic("SecurityAnalysisAPI")
CORS(app)

# Configure template directory
TEMPLATE_DIR = os.getenv("TEMPLATE_DIR", "templates")
os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Global analyzers dictionary
analyzers = {}
supported_log_types = ["crowdstrike", "cloudflare", "defender"]

# Configuration
MODEL_DIR = os.getenv("MODEL_DIR", "models")
API_VERSION = "v1"
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100000"))

@app.listener('before_server_start')
async def setup_analyzers(app, loop):
    """Load all trained models on server startup"""
    logger.info("🚀 Loading trained models...")
    
    global analyzers
    
    for log_type in supported_log_types:
        try:
            analyzer = create_analyzer(log_type, MODEL_DIR)
            
            if analyzer.load_model():
                analyzers[log_type] = analyzer
                logger.info(f"✅ Loaded {log_type} model")
            else:
                logger.warning(f"⚠️ Failed to load {log_type} model")
                
        except Exception as e:
            logger.error(f"❌ Error loading {log_type} model: {e}")
    
    if not analyzers:
        logger.error("❌ No models loaded! Server may not function properly.")
    else:
        logger.info(f"✅ Loaded {len(analyzers)} models: {list(analyzers.keys())}")

# ===== ROUTE HANDLERS =====

@app.route("/", methods=["GET"])
async def root_redirect(request: Request):
    """Redirect root to login page"""
    return redirect("/login")

@app.route("/login", methods=["GET"])
async def login_page(request: Request):
    """Serve the login page"""
    try:
        template_path = os.path.join(TEMPLATE_DIR, "login.html")
        
        if os.path.exists(template_path):
            return await file(template_path)
        else:
            # Return embedded HTML if template file doesn't exist
            return html(get_login_html())
            
    except Exception as e:
        logger.error(f"Error serving login page: {e}")
        return html(get_login_html())

@app.route("/dashboard", methods=["GET"])
async def dashboard_page(request: Request):
    """Serve the analysis dashboard"""
    try:
        template_path = os.path.join(TEMPLATE_DIR, "dashboard.html")
        
        if os.path.exists(template_path):
            return await file(template_path)
        else:
            # Return embedded HTML if template file doesn't exist
            return html(get_dashboard_html())
            
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        return html(get_dashboard_html())

def get_login_html():
    """Return embedded login HTML"""
    return """
    <!DOCTYPE html>
    <html><head><title>Login - Security Analyzer</title></head>
    <body>
    <h1>Security Analyzer - Login</h1>
    <p>Please save the login.html file to the templates/ directory.</p>
    <p><a href="/dashboard">Go to Dashboard</a></p>
    </body></html>
    """

def get_dashboard_html():
    """Return embedded dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html><head><title>Dashboard - Security Analyzer</title></head>
    <body>
    <h1>Security Analyzer - Dashboard</h1>
    <p>Please save the dashboard.html file to the templates/ directory.</p>
    <p>API Status: <span id="status">Loading...</span></p>
    <script>
    fetch('/api/v1/status')
        .then(r => r.json())
        .then(data => {
            document.getElementById('status').textContent = 'Online - Models: ' + Object.keys(data.models || {}).join(', ');
        })
        .catch(() => {
            document.getElementById('status').textContent = 'Offline';
        });
    </script>
    </body></html>
    """

@app.route("/health", methods=["GET"])
async def health_check(request: Request):
    """Health check endpoint for monitoring"""
    return sanic_json({
        "status": "healthy",
        "service": "Security Analysis API",
        "version": API_VERSION,
        "timestamp": datetime.now().isoformat(),
        "loaded_models": list(analyzers.keys())
    })

@app.route("/api/v1/status", methods=["GET"])
async def get_status(request: Request):
    """Get API status and model information"""
    model_info = {}
    
    for log_type, analyzer in analyzers.items():
        model_info[log_type] = analyzer.get_model_info()
    
    return sanic_json({
        "status": "operational",
        "api_version": API_VERSION,
        "models": model_info,
        "supported_log_types": supported_log_types,
        "max_batch_size": MAX_BATCH_SIZE,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/v1/upload", methods=["POST"])
async def upload_and_analyze(request: Request):
    """Upload JSON file and analyze automatically"""
    
    try:
        # Check if file was uploaded
        if not request.files:
            raise InvalidUsage("No file uploaded")
        
        # Get the uploaded file
        uploaded_file = None
        for field_name in request.files:
            uploaded_file = request.files[field_name][0]  # Get first file
            break
        
        if not uploaded_file:
            raise InvalidUsage("No file found in request")
        
        # Validate file type
        if not uploaded_file.name.endswith('.json'):
            raise InvalidUsage("Only JSON files are supported")
        
        # Parse JSON content
        try:
            file_content = uploaded_file.body.decode('utf-8')
            json_data = json.loads(file_content)
        except json.JSONDecodeError as e:
            raise InvalidUsage(f"Invalid JSON file: {str(e)}")
        except UnicodeDecodeError:
            raise InvalidUsage("File encoding not supported. Please use UTF-8 encoded JSON files")
        
        # Get log type from request (default to crowdstrike)
        log_type = request.form.get("log_type", ["crowdstrike"])[0]
        
        if log_type not in supported_log_types and log_type != "mixed":
            raise InvalidUsage(f"Unsupported log type: {log_type}")
        
        # Determine analysis type and endpoint
        if log_type == "mixed":
            # Mixed log analysis
            if not isinstance(json_data, dict):
                raise InvalidUsage("Mixed analysis requires object with log type keys")
            
            return await analyze_mixed_logs_data(json_data)
            
        else:
            # Single log type analysis
            if log_type not in analyzers:
                raise ServerError(f"Model for {log_type} not loaded")
            
            analyzer = analyzers[log_type]
            
            if isinstance(json_data, list):
                # Batch analysis
                if len(json_data) > MAX_BATCH_SIZE:
                    raise InvalidUsage(f"Batch size {len(json_data)} exceeds maximum {MAX_BATCH_SIZE}")
                
                results = analyzer.batch_predict(json_data)
                
                # Calculate summary
                successful = [r for r in results if 'error' not in r]
                failed = [r for r in results if 'error' in r]
                
                if successful:
                    attack_probs = [r['attack_probability'] for r in successful]
                    avg_attack_prob = sum(attack_probs) / len(attack_probs)
                    
                    risk_counts = {}
                    for result in successful:
                        risk_level = result.get('risk_level', 'UNKNOWN')
                        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
                else:
                    avg_attack_prob = 0.0
                    risk_counts = {}
                
                return sanic_json({
                    "success": True,
                    "analysis_type": "batch",
                    "predictions": results,
                    "summary": {
                        "total_sessions": len(json_data),
                        "successful_predictions": len(successful),
                        "failed_predictions": len(failed),
                        "average_attack_probability": round(avg_attack_prob, 4),
                        "risk_distribution": risk_counts
                    },
                    "model_type": log_type,
                    "file_info": {
                        "filename": uploaded_file.name,
                        "size_bytes": len(uploaded_file.body)
                    },
                    "timestamp": datetime.now().isoformat()
                })
                
            else:
                # Single session analysis
                result = analyzer.predict_session(json_data)
                
                return sanic_json({
                    "success": True,
                    "analysis_type": "single",
                    "prediction": result,
                    "model_type": log_type,
                    "file_info": {
                        "filename": uploaded_file.name,
                        "size_bytes": len(uploaded_file.body)
                    },
                    "timestamp": datetime.now().isoformat()
                })
        
    except InvalidUsage:
        raise
    except Exception as e:
        logger.error(f"Upload and analyze error: {e}")
        logger.error(traceback.format_exc())
        raise ServerError(f"Analysis failed: {str(e)}")

@app.route("/api/v1/predict/<log_type>", methods=["POST"])
async def predict_single(request: Request, log_type: str):
    """Predict attack probability for a single session"""
    
    # Validate log type
    if log_type not in supported_log_types:
        raise InvalidUsage(f"Unsupported log type: {log_type}. Supported types: {supported_log_types}")
    
    # Check if model is loaded
    if log_type not in analyzers:
        raise ServerError(f"Model for {log_type} not loaded")
    
    try:
        # Parse request body
        if not request.json:
            raise InvalidUsage("Request body must be valid JSON")
        
        session_data = request.json
        
        # Validate required fields
        if not isinstance(session_data, dict):
            raise InvalidUsage("Request body must be a JSON object")
        
        # Get analyzer and make prediction
        analyzer = analyzers[log_type]
        result = analyzer.predict_session(session_data)
        
        return sanic_json({
            "success": True,
            "prediction": result,
            "model_type": log_type,
            "api_version": API_VERSION,
            "timestamp": datetime.now().isoformat()
        })
        
    except InvalidUsage:
        raise
    except Exception as e:
        logger.error(f"Prediction error for {log_type}: {e}")
        logger.error(traceback.format_exc())
        raise ServerError(f"Prediction failed: {str(e)}")

@app.route("/api/v1/predict/<log_type>/batch", methods=["POST"])
async def predict_batch(request: Request, log_type: str):
    """Predict attack probabilities for multiple sessions"""
    
    # Validate log type
    if log_type not in supported_log_types:
        raise InvalidUsage(f"Unsupported log type: {log_type}. Supported types: {supported_log_types}")
    
    # Check if model is loaded
    if log_type not in analyzers:
        raise ServerError(f"Model for {log_type} not loaded")
    
    try:
        # Parse request body
        if not request.json:
            raise InvalidUsage("Request body must be valid JSON")
        
        request_data = request.json
        
        # Extract sessions list
        if isinstance(request_data, list):
            sessions = request_data
        elif isinstance(request_data, dict) and "sessions" in request_data:
            sessions = request_data["sessions"]
        else:
            raise InvalidUsage("Request must contain 'sessions' array or be an array of sessions")
        
        # Validate batch size
        if len(sessions) > MAX_BATCH_SIZE:
            raise InvalidUsage(f"Batch size {len(sessions)} exceeds maximum {MAX_BATCH_SIZE}")
        
        # Validate sessions
        if not isinstance(sessions, list):
            raise InvalidUsage("Sessions must be an array")
        
        for i, session in enumerate(sessions):
            if not isinstance(session, dict):
                raise InvalidUsage(f"Session {i} must be a JSON object")
        
        # Get analyzer and make predictions
        analyzer = analyzers[log_type]
        results = analyzer.batch_predict(sessions)
        
        # Calculate summary statistics
        successful_predictions = [r for r in results if 'error' not in r]
        failed_predictions = [r for r in results if 'error' in r]
        
        if successful_predictions:
            attack_probs = [r['attack_probability'] for r in successful_predictions]
            avg_attack_prob = sum(attack_probs) / len(attack_probs)
            
            risk_counts = {}
            for result in successful_predictions:
                risk_level = result.get('risk_level', 'UNKNOWN')
                risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        else:
            avg_attack_prob = 0.0
            risk_counts = {}
        
        return sanic_json({
            "success": True,
            "predictions": results,
            "summary": {
                "total_sessions": len(sessions),
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(failed_predictions),
                "average_attack_probability": round(avg_attack_prob, 4),
                "risk_distribution": risk_counts
            },
            "model_type": log_type,
            "api_version": API_VERSION,
            "timestamp": datetime.now().isoformat()
        })
        
    except InvalidUsage:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error for {log_type}: {e}")
        logger.error(traceback.format_exc())
        raise ServerError(f"Batch prediction failed: {str(e)}")

@app.route("/api/v1/analyze", methods=["POST"])
async def analyze_mixed_logs(request: Request):
    """Analyze logs from multiple sources automatically"""
    
    try:
        # Parse request body
        if not request.json:
            raise InvalidUsage("Request body must be valid JSON")
        
        request_data = request.json
        
        # Validate request structure
        if not isinstance(request_data, dict):
            raise InvalidUsage("Request body must be a JSON object")
        
        return await analyze_mixed_logs_data(request_data)
        
    except InvalidUsage:
        raise
    except Exception as e:
        logger.error(f"Mixed log analysis error: {e}")
        logger.error(traceback.format_exc())
        raise ServerError(f"Analysis failed: {str(e)}")

async def analyze_mixed_logs_data(json_data: dict):
    """Helper function for mixed log analysis"""
    results = {}
    total_processed = 0
    
    # Process each log type
    for log_type in supported_log_types:
        if log_type in json_data and log_type in analyzers:
            sessions = json_data[log_type]
            
            if not isinstance(sessions, list):
                results[log_type] = {
                    "error": f"Data for {log_type} must be an array",
                    "processed": 0
                }
                continue
            
            if len(sessions) > MAX_BATCH_SIZE:
                results[log_type] = {
                    "error": f"Batch size {len(sessions)} exceeds maximum {MAX_BATCH_SIZE}",
                    "processed": 0
                }
                continue
            
            # Make predictions
            analyzer = analyzers[log_type]
            predictions = analyzer.batch_predict(sessions)
            
            # Calculate summary
            successful = [p for p in predictions if 'error' not in p]
            failed = [p for p in predictions if 'error' in p]
            
            if successful:
                attack_probs = [p['attack_probability'] for p in successful]
                avg_attack_prob = sum(attack_probs) / len(attack_probs)
                
                risk_counts = {}
                for pred in successful:
                    risk_level = pred.get('risk_level', 'UNKNOWN')
                    risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            else:
                avg_attack_prob = 0.0
                risk_counts = {}
            
            results[log_type] = {
                "predictions": predictions,
                "summary": {
                    "total_sessions": len(sessions),
                    "successful_predictions": len(successful),
                    "failed_predictions": len(failed),
                    "average_attack_probability": round(avg_attack_prob, 4),
                    "risk_distribution": risk_counts
                }
            }
            
            total_processed += len(sessions)
    
    if not results:
        raise InvalidUsage("No valid log data provided for supported types: " + str(supported_log_types))
    
    return sanic_json({
        "success": True,
        "analysis_type": "mixed",
        "results": results,
        "global_summary": {
            "total_sessions_processed": total_processed,
            "log_types_processed": list(results.keys())
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/v1/models/reload", methods=["POST"])
async def reload_models(request: Request):
    """Reload all models (admin endpoint)"""
    
    try:
        global analyzers
        old_count = len(analyzers)
        analyzers.clear()
        
        # Reload all models
        for log_type in supported_log_types:
            try:
                analyzer = create_analyzer(log_type, MODEL_DIR)
                
                if analyzer.load_model():
                    analyzers[log_type] = analyzer
                    logger.info(f"✅ Reloaded {log_type} model")
                else:
                    logger.warning(f"⚠️ Failed to reload {log_type} model")
                    
            except Exception as e:
                logger.error(f"❌ Error reloading {log_type} model: {e}")
        
        new_count = len(analyzers)
        
        return sanic_json({
            "success": True,
            "message": f"Reloaded models: {old_count} -> {new_count}",
            "loaded_models": list(analyzers.keys()),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise ServerError(f"Model reload failed: {str(e)}")

# ===== ERROR HANDLERS =====

@app.exception(InvalidUsage)
async def handle_invalid_usage(request: Request, exception: InvalidUsage):
    """Handle invalid usage errors"""
    return sanic_json({
        "success": False,
        "error": "Invalid request",
        "message": str(exception),
        "timestamp": datetime.now().isoformat()
    }, status=400)

@app.exception(ServerError)
async def handle_server_error(request: Request, exception: ServerError):
    """Handle server errors"""
    return sanic_json({
        "success": False,
        "error": "Server error",
        "message": str(exception),
        "timestamp": datetime.now().isoformat()
    }, status=500)

@app.exception(Exception)
async def handle_generic_exception(request: Request, exception: Exception):
    """Handle any other exceptions"""
    logger.error(f"Unhandled exception: {exception}")
    logger.error(traceback.format_exc())
    
    return sanic_json({
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": datetime.now().isoformat()
    }, status=500)

# ===== MIDDLEWARE =====

@app.middleware('request')
async def log_request(request):
    """Log incoming requests"""
    logger.info(f"📥 {request.method} {request.path} from {request.ip}")

@app.middleware('response')
async def log_response(request, response):
    """Log responses"""
    logger.info(f"📤 {request.method} {request.path} -> {response.status}")

# ===== MAIN =====

if __name__ == "__main__":
    # Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    WORKERS = int(os.getenv("WORKERS", "1"))
    
    print("🚀 Starting Security Analysis API Server")
    print("=" * 50)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Workers: {WORKERS}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Supported Log Types: {supported_log_types}")
    print("=" * 50)
    print("📄 Pages available:")
    print(f"  🔐 Login: http://{HOST}:{PORT}/login")
    print(f"  📊 Dashboard: http://{HOST}:{PORT}/dashboard")
    print(f"  🏥 Health: http://{HOST}:{PORT}/health")
    print("=" * 50)
    
    try:
        app.run(
            host=HOST,
            port=PORT,
            debug=DEBUG,
            workers=WORKERS,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)