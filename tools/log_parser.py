"""
Training Log Parser - Handles CSV, JSON, YAML formats
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import uuid


class TrainingLogParser:
    def __init__(self):
        # Store parsed logs in memory (log_id -> DataFrame)
        self._logs: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Dict] = {}
    
    def parse(self, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """
        Parse training log file and return summary
        
        Args:
            file_path: Path to the log file
            format: File format (csv, json, yaml, auto)
            
        Returns:
            Dictionary with log_id and metadata
        """
        # Detect format if auto
        if format == "auto":
            format = self._detect_format(file_path)
        
        # Parse based on format
        if format == "csv":
            df = self._parse_csv(file_path)
        elif format == "json":
            df = self._parse_json(file_path)
        elif format == "yaml":
            df = self._parse_yaml(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Generate unique log ID
        log_id = str(uuid.uuid4())[:8]
        
        # Store in memory
        self._logs[log_id] = df
        self._metadata[log_id] = {
            "file_path": file_path,
            "format": format,
            "total_epochs": len(df)
        }
        
        # Return summary
        return {
            "log_id": log_id,
            "total_epochs": len(df),
            "metrics_available": list(df.columns),
            "format": format,
            "file_path": file_path,
            "sample_data": df.head(3).to_dict('records')
        }
    
    def get_log(self, log_id: str) -> Optional[pd.DataFrame]:
        """Retrieve parsed log by ID"""
        return self._logs.get(log_id)
    
    def get_metadata(self, log_id: str) -> Optional[Dict]:
        """Get metadata for a log"""
        return self._metadata.get(log_id)
    
    def _detect_format(self, file_path: str) -> str:
        """Auto-detect file format from extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.csv':
            return 'csv'
        elif ext == '.json':
            return 'json'
        elif ext in ['.yaml', '.yml']:
            return 'yaml'
        else:
            raise ValueError(f"Cannot detect format from extension: {ext}")
    
    def _parse_csv(self, file_path: str) -> pd.DataFrame:
        """Parse CSV training log"""
        try:
            df = pd.read_csv(file_path)
            
            # Validate required columns
            if 'epoch' not in df.columns:
                # If no epoch column, create one
                df.insert(0, 'epoch', range(len(df)))
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {str(e)}")
    
    def _parse_json(self, file_path: str) -> pd.DataFrame:
        """Parse JSON training log"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if 'epochs' in data:
                # Format: {"epochs": [{epoch: 1, ...}, {epoch: 2, ...}]}
                df = pd.DataFrame(data['epochs'])
            elif isinstance(data, list):
                # Format: [{epoch: 1, ...}, {epoch: 2, ...}]
                df = pd.DataFrame(data)
            else:
                raise ValueError("Unexpected JSON structure")
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {str(e)}")
    
    def _parse_yaml(self, file_path: str) -> pd.DataFrame:
        """Parse YAML config (returns config as single-row DataFrame)"""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Flatten nested dict for easier access
            flattened = self._flatten_dict(data)
            df = pd.DataFrame([flattened])
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {str(e)}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)