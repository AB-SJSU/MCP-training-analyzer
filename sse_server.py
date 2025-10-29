#!/usr/bin/env python3
"""
SSE-based MCP Server for ChatGPT Connector - FIXED
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent, ImageContent

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response, JSONResponse
from starlette.requests import Request
import uvicorn

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

# Create MCP server
mcp_server = Server("training-log-analyzer")


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="analyze_training_scenario",
            description="Analyze a demo training scenario. Options: overfitting, divergence, perfect, slow",
            inputSchema={
                "type": "object",
                "properties": {
                    "scenario": {
                        "type": "string",
                        "enum": ["overfitting", "divergence", "perfect", "slow"],
                        "description": "Demo scenario to analyze"
                    }
                },
                "required": ["scenario"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "analyze_training_scenario":
            return await analyze_scenario(arguments["scenario"])
        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e)})
        )]


async def analyze_scenario(scenario: str) -> list[TextContent]:
    """Analyze training scenario"""
    
    scenario_files = {
        'overfitting': 'demo_data/overfitting.csv',
        'divergence': 'demo_data/divergence.csv',
        'perfect': 'demo_data/perfect_training.csv',
        'slow': 'demo_data/slow_convergence.csv',
    }
    
    file_path = scenario_files.get(scenario)
    if not file_path:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Parse
    parse_result = parser.parse(file_path)
    log_id = parse_result['log_id']
    
    # Detect issues
    df = parser.get_log(log_id)
    issues = detector.detect_all(df)
    
    # Generate suggestions
    suggestions = generate_suggestions(issues)
    
    # Format response
    response = f"""# Training Analysis: {scenario.upper()}

**Health Score:** {issues['health_score']}/100
**Issues Found:** {issues['total_issues']}

"""
    
    if issues['issues_detected']:
        response += "## Problems Detected:\n\n"
        for idx, issue in enumerate(issues['issues_detected'], 1):
            response += f"{idx}. **{issue['type'].upper()}** ({issue['severity']})\n"
            response += f"   {issue['description']}\n\n"
    
    if suggestions['suggestions']:
        response += "## Recommendations:\n\n"
        for idx, sug in enumerate(suggestions['suggestions'][:3], 1):
            response += f"{idx}. **{sug['parameter']}**: {sug['suggested_value']}\n"
            response += f"   {sug['reason']}\n\n"
    
    return [TextContent(type="text", text=response)]


def generate_suggestions(issues_result: dict) -> dict:
    """Generate hyperparameter suggestions"""
    suggestions = []
    issues = issues_result.get("issues_detected", [])
    
    for issue in issues:
        issue_type = issue.get("type")
        
        if issue_type == "overfitting":
            suggestions.extend([
                {
                    "parameter": "dropout",
                    "suggested_value": "0.3-0.5",
                    "priority": "high",
                    "reason": "Add dropout to reduce overfitting"
                },
                {
                    "parameter": "weight_decay",
                    "suggested_value": "1e-4",
                    "priority": "high",
                    "reason": "L2 regularization prevents overfitting"
                }
            ])
        elif issue_type == "divergence":
            suggestions.append({
                "parameter": "learning_rate",
                "suggested_value": "reduce by 10x",
                "priority": "critical",
                "reason": "Training diverged - LR too high"
            })
    
    return {"suggestions": suggestions}


# HTTP Routes
async def health(request):
    return JSONResponse({"status": "healthy"})


async def root(request):
    return JSONResponse({
        "service": "Training Log Analyzer MCP",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "sse": "/sse"
        }
    })


async def handle_sse(request: Request):
    """Handle SSE endpoint - FIXED VERSION"""
    
    # Create transport (no async context manager)
    transport = SseServerTransport("/messages")
    
    # Handle the connection
    async def run_server():
        try:
            # Initialize the transport
            read_stream, write_stream = transport.connect_streams()
            
            # Run the MCP server
            init_options = mcp_server.create_initialization_options()
            await mcp_server.run(read_stream, write_stream, init_options)
            
        except Exception as e:
            print(f"Error in MCP server: {e}")
    
    # Start server task
    asyncio.create_task(run_server())
    
    # Return SSE response
    return transport.get_response()


# Create Starlette app
app = Starlette(
    debug=True,
    routes=[
        Route('/', root),
        Route('/health', health),
        Route('/sse', handle_sse),
    ]
)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting MCP SSE Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)