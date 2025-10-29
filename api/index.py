#!/usr/bin/env python3
"""
Vercel serverless entry point for FastAPI app
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Import the FastAPI app from http_server.py
from http_server import app

# Export for Vercel
handler = app