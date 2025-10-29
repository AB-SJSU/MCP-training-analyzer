#!/usr/bin/env python3
"""
MCP Server for Training Log Analysis - Complete with 6 tools
"""

from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
import json
import sys
import os
from pathlib import Path

# Import all tools
sys.path.append(str(Path(__file__).parent))
from tools.log_parser import TrainingLogParser
from tools.issue_detector import IssueDetector
from tools.metrics_analyzer import MetricsAnalyzer
from tools.comparator import EpochComparator
from tools.visualizer import TrainingVisualizer

# Initialize all tools
parser = TrainingLogParser()
detector = IssueDetector()
analyzer = MetricsAnalyzer()
comparator = EpochComparator()
visualizer = TrainingVisualizer()

# Create MCP server
app = Server("training-log-analyzer")

PORT = int(os.getenv("PORT", 8000))

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Define all 6 tools"""
    return [
        Tool(
            name="parse_training_log",
            description="Parse a machine learning training log file (CSV, JSON, or YAML) and extract metrics like loss, accuracy, learning rate",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the training log file (e.g., demo_data/overfitting.csv)"
                    },
                    "format": {
                        "type": "string",
                        "enum": ["csv", "json", "yaml", "auto"],
                        "description": "File format (default: auto-detect)",
                        "default": "auto"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="detect_training_issues",
            description="Analyze training dynamics to detect overfitting, divergence, plateau, or slow convergence issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "ID of parsed training log (returned from parse_training_log)"
                    }
                },
                "required": ["log_id"]
            }
        ),
        Tool(
            name="get_metrics_summary",
            description="Calculate comprehensive statistics for training metrics including best/worst epochs, trends, and convergence analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "ID of parsed training log"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific metrics to analyze (optional, defaults to all)"
                    }
                },
                "required": ["log_id"]
            }
        ),
        Tool(
            name="compare_epochs",
            description="Compare training metrics between specified epochs or across different training runs",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "ID of parsed training log"
                    },
                    "epoch_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "[start_epoch, end_epoch] or null for auto"
                    },
                    "comparison_type": {
                        "type": "string",
                        "enum": ["sequential", "interval", "best_vs_worst"],
                        "description": "Type of comparison to perform",
                        "default": "sequential"
                    }
                },
                "required": ["log_id"]
            }
        ),
        Tool(
            name="suggest_hyperparameters",
            description="Suggest hyperparameter improvements based on detected training issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "ID of parsed training log"
                    }
                },
                "required": ["log_id"]
            }
        ),
        Tool(
            name="generate_visualization",
            description="Generate visualization of training metrics (loss curves, accuracy curves, learning rate schedule, or custom comparison)",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "ID of parsed training log"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["loss_curve", "accuracy_curve", "learning_rate_schedule", "comparison"],
                        "description": "Type of visualization to generate"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Metrics to visualize (for comparison plots)"
                    }
                },
                "required": ["log_id", "plot_type"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list:
    """Handle tool execution"""
    
    try:
        if name == "parse_training_log":
            result = parser.parse(
                arguments["file_path"], 
                arguments.get("format", "auto")
            )
            return [TextContent(
                type="text", 
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "detect_training_issues":
            log_id = arguments["log_id"]
            df = parser.get_log(log_id)
            
            if df is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Log {log_id} not found"})
                )]
            
            result = detector.detect_all(df)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "get_metrics_summary":
            log_id = arguments["log_id"]
            df = parser.get_log(log_id)
            
            if df is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Log {log_id} not found"})
                )]
            
            metrics = arguments.get("metrics")
            result = analyzer.get_summary(df, metrics)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "compare_epochs":
            log_id = arguments["log_id"]
            df = parser.get_log(log_id)
            
            if df is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Log {log_id} not found"})
                )]
            
            epoch_range = arguments.get("epoch_range")
            comparison_type = arguments.get("comparison_type", "sequential")
            
            result = comparator.compare(df, epoch_range, comparison_type)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        elif name == "suggest_hyperparameters":
            log_id = arguments["log_id"]
            df = parser.get_log(log_id)
            
            if df is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Log {log_id} not found"})
                )]
            
            # Detect issues first
            issues = detector.detect_all(df)
            
            # Generate suggestions
            suggestions = generate_suggestions(issues)
            
            return [TextContent(
                type="text",
                text=json.dumps(suggestions, indent=2)
            )]
        
        elif name == "generate_visualization":
            log_id = arguments["log_id"]
            df = parser.get_log(log_id)
            
            if df is None:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Log {log_id} not found"})
                )]
            
            plot_type = arguments["plot_type"]
            metrics = arguments.get("metrics")
            
            result = visualizer.generate(df, plot_type, metrics)
            
            # Return both text description and image
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "description": result.get("description"),
                        "plot_type": result.get("plot_type"),
                        "note": "Image data included as base64"
                    }, indent=2)
                ),
                ImageContent(
                    type="image",
                    data=result["image_base64"],
                    mimeType="image/png"
                )
            ]
        
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    
    except Exception as e:
        import traceback
        return [TextContent(
            type="text",
            text=json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
        )]


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
                },
                {
                    "parameter": "learning_rate",
                    "current_value": "unknown",
                    "suggested_value": "reduce by 50%",
                    "priority": "medium",
                    "reason": "Lower learning rate for more stable training"
                }
            ])
        
        elif issue_type == "divergence":
            epoch = issue.get("epoch", "unknown")
            suggestions.extend([
                {
                    "parameter": "learning_rate",
                    "current_value": "too high",
                    "suggested_value": "reduce by 10x",
                    "priority": "critical",
                    "reason": f"Training diverged at epoch {epoch}. Learning rate too high."
                },
                {
                    "parameter": "gradient_clipping",
                    "current_value": "not set",
                    "suggested_value": "1.0",
                    "priority": "high",
                    "reason": "Clip gradients to prevent explosion"
                }
            ])
        
        elif issue_type == "plateau":
            suggestions.extend([
                {
                    "parameter": "learning_rate_schedule",
                    "current_value": "constant",
                    "suggested_value": "reduce on plateau",
                    "priority": "high",
                    "reason": "Reduce LR when validation metric plateaus"
                },
                {
                    "parameter": "early_stopping",
                    "current_value": "not set",
                    "suggested_value": "patience=10",
                    "priority": "medium",
                    "reason": "Stop training when no improvement"
                }
            ])
        
        elif issue_type == "slow_convergence":
            suggestions.extend([
                {
                    "parameter": "learning_rate",
                    "current_value": "too low",
                    "suggested_value": "increase by 2-5x",
                    "priority": "high",
                    "reason": "Training too slow. Increase learning rate."
                },
                {
                    "parameter": "batch_size",
                    "current_value": "unknown",
                    "suggested_value": "try larger batch",
                    "priority": "medium",
                    "reason": "Larger batch size may speed up convergence"
                }
            ])
    
    return {
        "suggestions": suggestions,
        "total_suggestions": len(suggestions),
        "summary": f"Generated {len(suggestions)} hyperparameter recommendations based on detected issues."
    }



if __name__ == "__main__":
    import asyncio
    import mcp.server.stdio

    # For HTTP deployment, we need a wrapper
    # Render will run this with stdio, which is what MCP uses

    async def main():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream, 
                write_stream, 
                app.create_initialization_options()
            )

    asyncio.run(main())