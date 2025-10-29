#!/usr/bin/env python3
"""
MCP Server with SSE transport for ChatGPT Connector
"""

import asyncio
import json
import sys
from pathlib import Path
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import Tool, TextContent
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
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
            description="Analyze a demo training scenario (overfitting, divergence, perfect, or slow). Returns comprehensive analysis with issues detected and recommendations.",
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
        ),
        Tool(
            name="get_detailed_metrics",
            description="Get detailed metrics summary for a previously analyzed log",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "Log ID from previous analysis"
                    }
                },
                "required": ["log_id"]
            }
        ),
        Tool(
            name="visualize_training_curves",
            description="Generate visualization of training metrics",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_id": {
                        "type": "string",
                        "description": "Log ID from previous analysis"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["loss_curve", "accuracy_curve", "learning_rate_schedule"],
                        "description": "Type of visualization"
                    }
                },
                "required": ["log_id", "plot_type"]
            }
        )
    ]


@mcp_server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "analyze_training_scenario":
            return await analyze_scenario(arguments["scenario"])
        
        elif name == "get_detailed_metrics":
            return await get_metrics(arguments["log_id"])
        
        elif name == "visualize_training_curves":
            return await visualize(arguments["log_id"], arguments["plot_type"])
        
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
    """Analyze training scenario - comprehensive analysis"""
    
    scenario_files = {
        'overfitting': 'demo_data/overfitting.csv',
        'divergence': 'demo_data/divergence.csv',
        'perfect': 'demo_data/perfect_training.csv',
        'slow': 'demo_data/slow_convergence.csv',
    }
    
    file_path = scenario_files.get(scenario)
    if not file_path:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Parse log
    parse_result = parser.parse(file_path)
    log_id = parse_result['log_id']
    
    # Get dataframe
    df = parser.get_log(log_id)
    
    # Detect issues
    issues = detector.detect_all(df)
    
    # Generate suggestions
    suggestions = generate_suggestions(issues)
    
    # Format response
    response = f"""# Training Log Analysis: {scenario.upper()}

**Log ID:** {log_id} *(save this for detailed analysis)*
**Total Epochs:** {parse_result['total_epochs']}
**Health Score:** {issues['health_score']}/100

## Issues Detected: {issues['total_issues']}

"""
    
    if issues['issues_detected']:
        for idx, issue in enumerate(issues['issues_detected'], 1):
            response += f"""### {idx}. {issue['type'].upper()} ({issue['severity']} severity)
{issue['description']}

"""
    else:
        response += "✅ No major issues detected! Training looks healthy.\n\n"
    
    if suggestions['suggestions']:
        response += "## Recommendations\n\n"
        for idx, sug in enumerate(suggestions['suggestions'], 1):
            response += f"""**{idx}. {sug['parameter']}** (Priority: {sug['priority']})
- Current: {sug['current_value']}
- Suggested: {sug['suggested_value']}
- Reason: {sug['reason']}

"""
    
    response += f"\n---\n**Next steps:** Use log_id `{log_id}` to get detailed metrics or visualizations."
    
    return [TextContent(type="text", text=response)]


async def get_metrics(log_id: str) -> list[TextContent]:
    """Get detailed metrics"""
    
    df = parser.get_log(log_id)
    if df is None:
        raise ValueError(f"Log {log_id} not found")
    
    summary = analyzer.get_summary(df)
    
    response = f"""# Detailed Metrics Summary

**Total Epochs:** {summary['total_epochs']}
**Best Epoch:** {summary['best_epoch']['epoch']}
**Convergence Status:** {summary['convergence']['status']}

## Metrics Statistics

"""
    
    for metric, stats in summary['metrics'].items():
        response += f"""### {metric}
- Range: {stats['min']:.4f} to {stats['max']:.4f}
- Mean: {stats['mean']:.4f}
- Trend: {stats['trend']}

"""
    
    return [TextContent(type="text", text=response)]


async def visualize(log_id: str, plot_type: str) -> list[TextContent]:
    """Generate visualization"""
    
    df = parser.get_log(log_id)
    if df is None:
        raise ValueError(f"Log {log_id} not found")
    
    result = visualizer.generate(df, plot_type, None)
    
    return [
        TextContent(type="text", text=f"Generated {plot_type} for log {log_id}"),
        TextContent(type="text", text=f"Image data: {result['image_base64'][:100]}...")
    ]


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
                    "current_value": "not set",
                    "suggested_value": "0.3-0.5",
                    "priority": "high",
                    "reason": "Add dropout layers to reduce overfitting"
                },
                {
                    "parameter": "weight_decay",
                    "current_value": "not set",
                    "suggested_value": "1e-4",
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
                "reason": "Training diverged - learning rate too high"
            })
        
        elif issue_type == "plateau":
            suggestions.append({
                "parameter": "learning_rate_schedule",
                "current_value": "constant",
                "suggested_value": "reduce on plateau",
                "priority": "high",
                "reason": "Use learning rate decay when plateaued"
            })
        
        elif issue_type == "slow_convergence":
            suggestions.append({
                "parameter": "learning_rate",
                "current_value": "too low",
                "suggested_value": "increase by 2-5x",
                "priority": "high",
                "reason": "Training too slow - increase learning rate"
            })
    
    return {
        "suggestions": suggestions,
        "total_suggestions": len(suggestions)
    }


# Starlette app for SSE
async def handle_sse(request):
    """Handle SSE connection from ChatGPT"""
    async with SseServerTransport("/messages") as transport:
        await mcp_server.run(
            transport.read_stream,
            transport.write_stream,
            mcp_server.create_initialization_options()
        )
    return Response()


async def handle_messages(request):
    """Handle message endpoint"""
    return Response("MCP Server Ready", media_type="text/plain")


app = Starlette(
    routes=[
        Route("/sse", handle_sse),
        Route("/messages", handle_messages, methods=["POST"]),
        Route("/", lambda r: Response("MCP Training Analyzer - SSE Server")),
        Route("/health", lambda r: Response('{"status":"healthy"}', media_type="application/json")),
    ]
)


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting MCP SSE Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)