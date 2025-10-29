from http.server import BaseHTTPRequestHandler
import json
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

from tools.log_parser import TrainingLogParser
from tools.issue_detector import IssueDetector
from tools.metrics_analyzer import MetricsAnalyzer
from tools.comparator import EpochComparator
from tools.visualizer import TrainingVisualizer
from http_server import generate_suggestions

# Initialize tools
parser = TrainingLogParser()
detector = IssueDetector()
analyzer = MetricsAnalyzer()
comparator = EpochComparator()
visualizer = TrainingVisualizer()


class handler(BaseHTTPRequestHandler):
    """MCP protocol handler for ChatGPT"""
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """List available tools"""
        if self.path == '/mcp' or self.path == '/mcp/':
            tools = {
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {
                            "name": "parse_training_log",
                            "description": "Parse ML training log file (CSV, JSON, YAML)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "file_path": {"type": "string"},
                                    "format": {"type": "string", "default": "auto"}
                                },
                                "required": ["file_path"]
                            }
                        },
                        {
                            "name": "detect_training_issues",
                            "description": "Detect overfitting, divergence, plateau issues",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "log_id": {"type": "string"}
                                },
                                "required": ["log_id"]
                            }
                        },
                        {
                            "name": "get_metrics_summary",
                            "description": "Calculate comprehensive metrics statistics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "log_id": {"type": "string"},
                                    "metrics": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["log_id"]
                            }
                        },
                        {
                            "name": "suggest_hyperparameters",
                            "description": "Suggest hyperparameter improvements",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "log_id": {"type": "string"}
                                },
                                "required": ["log_id"]
                            }
                        }
                    ]
                }
            }
            self._send_json(tools)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Execute tool"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            request = json.loads(body.decode('utf-8'))
            
            tool_name = request.get('name') or request.get('method')
            arguments = request.get('arguments') or request.get('params', {})
            
            result = self._execute_tool(tool_name, arguments)
            
            response = {
                "jsonrpc": "2.0",
                "id": request.get('id', 1),
                "result": result
            }
            
            self._send_json(response)
            
        except Exception as e:
            import traceback
            self._send_json({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": str(e),
                    "data": traceback.format_exc()
                }
            }, 500)
    
    def _execute_tool(self, name, arguments):
        """Execute the requested tool"""
        if name == "parse_training_log":
            return parser.parse(
                arguments["file_path"],
                arguments.get("format", "auto")
            )
        
        elif name == "detect_training_issues":
            df = parser.get_log(arguments["log_id"])
            if df is None:
                raise ValueError(f"Log {arguments['log_id']} not found")
            return detector.detect_all(df)
        
        elif name == "get_metrics_summary":
            df = parser.get_log(arguments["log_id"])
            if df is None:
                raise ValueError(f"Log {arguments['log_id']} not found")
            return analyzer.get_summary(df, arguments.get("metrics"))
        
        elif name == "suggest_hyperparameters":
            df = parser.get_log(arguments["log_id"])
            if df is None:
                raise ValueError(f"Log {arguments['log_id']} not found")
            issues = detector.detect_all(df)
            return generate_suggestions(issues)
        
        else:
            raise ValueError(f"Unknown tool: {name}")