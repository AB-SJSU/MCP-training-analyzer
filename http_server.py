#!/usr/bin/env python3
"""
HTTP Server for MCP Tools - Compatible with ChatGPT Apps
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import tools
sys.path.append(str(Path(__file__).parent))
from tools.log_parser import TrainingLogParser
from tools.issue_detector import IssueDetector
from tools.metrics_analyzer import MetricsAnalyzer
from tools.comparator import EpochComparator
from tools.visualizer import TrainingVisualizer

# Initialize tools
parser = TrainingLogParser()
detector = IssueDetector()
analyzer = MetricsAnalyzer()
comparator = EpochComparator()
visualizer = TrainingVisualizer()

# Create FastAPI app
app = FastAPI(
    title="Training Log Analyzer API",
    description="MCP-based tools for analyzing ML training logs",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class ParseRequest(BaseModel):
    file_path: str
    format: str = "auto"

class LogIDRequest(BaseModel):
    log_id: str

class MetricsRequest(BaseModel):
    log_id: str
    metrics: Optional[List[str]] = None

class CompareRequest(BaseModel):
    log_id: str
    epoch_range: Optional[List[int]] = None
    comparison_type: str = "sequential"

class VisualizeRequest(BaseModel):
    log_id: str
    plot_type: str
    metrics: Optional[List[str]] = None


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Training Log Analyzer API",
        "status": "running",
        "description": "MCP-based tools for analyzing ML training logs",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "tools": "/tools",
            "parse": "/tools/parse_training_log",
            "detect": "/tools/detect_training_issues",
            "metrics": "/tools/get_metrics_summary",
            "compare": "/tools/compare_epochs",
            "suggest": "/tools/suggest_hyperparameters",
            "visualize": "/tools/generate_visualization"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.get("/tools")
async def list_tools():
    """List available tools"""
    return {
        "tools": [
            {
                "name": "parse_training_log",
                "description": "Parse ML training log file (CSV, JSON, YAML)",
                "endpoint": "/tools/parse_training_log"
            },
            {
                "name": "detect_training_issues",
                "description": "Detect overfitting, divergence, plateau issues",
                "endpoint": "/tools/detect_training_issues"
            },
            {
                "name": "get_metrics_summary",
                "description": "Calculate comprehensive metrics statistics",
                "endpoint": "/tools/get_metrics_summary"
            },
            {
                "name": "compare_epochs",
                "description": "Compare metrics across epochs",
                "endpoint": "/tools/compare_epochs"
            },
            {
                "name": "suggest_hyperparameters",
                "description": "Suggest hyperparameter improvements",
                "endpoint": "/tools/suggest_hyperparameters"
            },
            {
                "name": "generate_visualization",
                "description": "Generate training visualizations",
                "endpoint": "/tools/generate_visualization"
            }
        ]
    }


@app.post("/tools/parse_training_log")
async def parse_training_log(request: ParseRequest):
    """Parse training log"""
    try:
        result = parser.parse(request.file_path, request.format)
        return result
    except Exception as e:
        logger.error(f"Error parsing log: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/detect_training_issues")
async def detect_training_issues(request: LogIDRequest):
    """Detect training issues"""
    try:
        df = parser.get_log(request.log_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        result = detector.detect_all(df)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error detecting issues: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/get_metrics_summary")
async def get_metrics_summary(request: MetricsRequest):
    """Get metrics summary"""
    try:
        df = parser.get_log(request.log_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        result = analyzer.get_summary(df, request.metrics)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/compare_epochs")
async def compare_epochs(request: CompareRequest):
    """Compare epochs"""
    try:
        df = parser.get_log(request.log_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        result = comparator.compare(df, request.epoch_range, request.comparison_type)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing epochs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/suggest_hyperparameters")
async def suggest_hyperparameters(request: LogIDRequest):
    """Suggest hyperparameters"""
    try:
        df = parser.get_log(request.log_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        # Detect issues
        issues = detector.detect_all(df)
        
        # Generate suggestions
        suggestions = generate_suggestions(issues)
        return suggestions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error suggesting hyperparameters: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/generate_visualization")
async def generate_visualization(request: VisualizeRequest):
    """Generate visualization"""
    try:
        df = parser.get_log(request.log_id)
        if df is None:
            raise HTTPException(status_code=404, detail=f"Log {request.log_id} not found")
        
        result = visualizer.generate(df, request.plot_type, request.metrics)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def generate_suggestions(issues_result: dict) -> dict:
    """Generate hyperparameter suggestions based on detected issues"""
    
    suggestions = []
    issues = issues_result.get("issues_detected", [])
    
    for issue in issues:
        issue_type = issue.get("type")
        
        if issue_type == "overfitting":
            suggestions.extend([
                {
                    "parameter": "dropout",
                    "current_value": "not set",
                    "suggested_value": "0.3-0.5",
                    "priority": "high",
                    "reason": "Add dropout layers to reduce overfitting"
                },
                {
                    "parameter": "weight_decay",
                    "current_value": "not set",
                    "suggested_value": "1e-4 to 1e-5",
                    "priority": "high",
                    "reason": "L2 regularization helps prevent overfitting"
                }
            ])
        
        elif issue_type == "divergence":
            suggestions.append({
                "parameter": "learning_rate",
                "current_value": "too high",
                "suggested_value": "reduce by 10x",
                "priority": "critical",
                "reason": f"Training diverged. Learning rate too high."
            })
        
        elif issue_type == "plateau":
            suggestions.append({
                "parameter": "learning_rate_schedule",
                "current_value": "constant",
                "suggested_value": "reduce on plateau",
                "priority": "high",
                "reason": "Reduce LR when validation metric plateaus"
            })
        
        elif issue_type == "slow_convergence":
            suggestions.append({
                "parameter": "learning_rate",
                "current_value": "too low",
                "suggested_value": "increase by 2-5x",
                "priority": "high",
                "reason": "Training too slow. Increase learning rate."
            })
    
    return {
        "suggestions": suggestions,
        "total_suggestions": len(suggestions),
        "summary": f"Generated {len(suggestions)} recommendations based on detected issues."
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"🚀 Starting Training Log Analyzer API on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)