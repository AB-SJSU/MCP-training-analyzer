# MCP Training Log Analyzer

An MCP server with 6 tools for analyzing machine learning training logs and detecting issues like overfitting and divergence.

## What It Does

- Parses training log files (CSV, JSON, YAML)
- Detects training issues (overfitting, divergence, plateau, slow convergence)
- Calculates metrics and statistics
- Compares epoch performance
- Suggests hyperparameter improvements
- Generates visualizations

## Tools

1. **parse_training_log** - Parse training data from files
2. **detect_training_issues** - Identify overfitting, divergence, etc.
3. **get_metrics_summary** - Get detailed statistics
4. **compare_epochs** - Compare performance across epochs
5. **suggest_hyperparameters** - Get actionable recommendations
6. **generate_visualization** - Create loss/accuracy curves

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-training-analyzer.git
cd mcp-training-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run HTTP Server
```bash
python http_server.py
# Server runs on http://localhost:8000
```

### Test with MCP Inspector
```bash
npx @modelcontextprotocol/inspector python