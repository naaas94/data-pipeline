"""
Health check endpoint for the ML pipeline.
Used by Kubernetes liveness and readiness probes.
"""

import os
import sys
import time
from datetime import datetime
from flask import Flask, jsonify
import threading

# Setup project paths for imports
from src.utils.path_setup import setup_project_paths
setup_project_paths()

from src.utils.logger import get_logger

app = Flask(__name__)
logger = get_logger("health_check")

# Global health status
health_status = {
    "status": "healthy",
    "timestamp": datetime.now().isoformat(),
    "startup_time": time.time(),
    "checks": {
        "pipeline_ready": False,
        "dependencies_available": False,
        "storage_accessible": False
    }
}

def check_pipeline_ready():
    """Check if the pipeline is ready to process data."""
    try:
        # Import pipeline components to check if they're available
        from src.pcc_pipeline import PCCDataPipeline
        from src.features.text_features import TextFeatureEngineer
        from src.features.embeddings import EmbeddingGenerator
        
        # Check if config file exists
        config_path = os.getenv('CONFIG_PATH', 'config.yaml')
        if not os.path.exists(config_path):
            return False
            
        # Try to initialize pipeline components
        pipeline = PCCDataPipeline(config_path)
        text_engineer = TextFeatureEngineer(pipeline.config)
        embedding_gen = EmbeddingGenerator(pipeline.config)
        
        health_status["checks"]["pipeline_ready"] = True
        return True
    except Exception as e:
        logger.error(f"Pipeline readiness check failed: {e}")
        health_status["checks"]["pipeline_ready"] = False
        return False

def check_dependencies():
    """Check if external dependencies are available."""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import sentence_transformers
        
        # Check if required directories exist
        required_dirs = ['output', 'logs', 'checkpoints', 'cache']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        
        health_status["checks"]["dependencies_available"] = True
        return True
    except Exception as e:
        logger.error(f"Dependencies check failed: {e}")
        health_status["checks"]["dependencies_available"] = False
        return False

def check_storage():
    """Check if storage is accessible."""
    try:
        # Test write access to output directory
        test_file = "output/health_check_test.txt"
        with open(test_file, 'w') as f:
            f.write(f"Health check test at {datetime.now().isoformat()}")
        
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        
        health_status["checks"]["storage_accessible"] = True
        return True
    except Exception as e:
        logger.error(f"Storage check failed: {e}")
        health_status["checks"]["storage_accessible"] = False
        return False

def update_health_status():
    """Update health status periodically."""
    while True:
        try:
            # Update timestamp
            health_status["timestamp"] = datetime.now().isoformat()
            
            # Run health checks
            pipeline_ok = check_pipeline_ready()
            deps_ok = check_dependencies()
            storage_ok = check_storage()
            
            # Update overall status
            if pipeline_ok and deps_ok and storage_ok:
                health_status["status"] = "healthy"
            else:
                health_status["status"] = "unhealthy"
            
            # Log status
            logger.info(f"Health status: {health_status['status']}", 
                       pipeline_ready=pipeline_ok,
                       dependencies_ok=deps_ok,
                       storage_ok=storage_ok)
            
            time.sleep(30)  # Update every 30 seconds
            
        except Exception as e:
            logger.error(f"Health check update failed: {e}")
            health_status["status"] = "error"
            time.sleep(30)

@app.route('/health')
def health():
    """Liveness probe endpoint."""
    return jsonify({
        "status": health_status["status"],
        "timestamp": health_status["timestamp"],
        "uptime": time.time() - health_status["startup_time"]
    })

@app.route('/ready')
def ready():
    """Readiness probe endpoint."""
    checks = health_status["checks"]
    all_checks_passed = all(checks.values())
    
    if all_checks_passed:
        return jsonify({
            "status": "ready",
            "timestamp": health_status["timestamp"],
            "checks": checks
        }), 200
    else:
        return jsonify({
            "status": "not_ready",
            "timestamp": health_status["timestamp"],
            "checks": checks
        }), 503

@app.route('/')
def root():
    """Root endpoint with basic info."""
    return jsonify({
        "service": "Privacy Intent Classification Pipeline",
        "version": "1.0.0",
        "status": health_status["status"],
        "timestamp": health_status["timestamp"]
    })

def start_health_server():
    """Start the health check server."""
    # Start health check thread
    health_thread = threading.Thread(target=update_health_status, daemon=True)
    health_thread.start()
    
    # Start Flask app
    port = int(os.getenv('HEALTH_PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == '__main__':
    start_health_server()
