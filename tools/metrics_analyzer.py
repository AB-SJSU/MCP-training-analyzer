"""
Metrics Analyzer - Calculate statistics and summaries
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class MetricsAnalyzer:
    def get_summary(self, df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary
        
        Args:
            df: Training log DataFrame
            metrics: List of metrics to analyze (None = all numeric columns)
            
        Returns:
            Dictionary with statistics
        """
        # If no metrics specified, use all numeric columns except epoch
        if metrics is None:
            metrics = [col for col in df.columns if col != 'epoch' and pd.api.types.is_numeric_dtype(df[col])]
        
        summary = {
            "total_epochs": len(df),
            "metrics": {}
        }
        
        for metric in metrics:
            if metric not in df.columns:
                continue
            
            values = df[metric].dropna()
            
            summary["metrics"][metric] = {
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std()),
                "first": float(values.iloc[0]),
                "last": float(values.iloc[-1]),
                "trend": self._calculate_trend(values)
            }
        
        # Add best/worst epochs
        summary["best_epoch"] = self.get_best_epoch(df, metrics)
        summary["worst_epoch"] = self.get_worst_epoch(df, metrics)
        
        # Add convergence info
        summary["convergence"] = self._analyze_convergence(df, metrics)
        
        return summary
    
    def get_best_epoch(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Find epoch with best performance"""
        
        # Prioritize validation accuracy, then validation loss
        if 'val_accuracy' in df.columns:
            best_idx = df['val_accuracy'].idxmax()
            metric_name = 'val_accuracy'
            metric_value = float(df['val_accuracy'].iloc[best_idx])
        elif 'val_loss' in df.columns:
            best_idx = df['val_loss'].idxmin()
            metric_name = 'val_loss'
            metric_value = float(df['val_loss'].iloc[best_idx])
        elif 'train_accuracy' in df.columns:
            best_idx = df['train_accuracy'].idxmax()
            metric_name = 'train_accuracy'
            metric_value = float(df['train_accuracy'].iloc[best_idx])
        else:
            return {"epoch": None, "metric": None, "value": None}
        
        return {
            "epoch": int(df['epoch'].iloc[best_idx]),
            "metric": metric_name,
            "value": metric_value,
            "details": df.iloc[best_idx].to_dict()
        }
    
    def get_worst_epoch(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Find epoch with worst performance"""
        
        # Skip first 5 epochs (initialization)
        df_subset = df.iloc[5:]
        
        if 'val_loss' in df_subset.columns:
            worst_idx = df_subset['val_loss'].idxmax()
            metric_name = 'val_loss'
            metric_value = float(df_subset['val_loss'].loc[worst_idx])
        elif 'val_accuracy' in df_subset.columns:
            worst_idx = df_subset['val_accuracy'].idxmin()
            metric_name = 'val_accuracy'
            metric_value = float(df_subset['val_accuracy'].loc[worst_idx])
        else:
            return {"epoch": None, "metric": None, "value": None}
        
        return {
            "epoch": int(df['epoch'].loc[worst_idx]),
            "metric": metric_name,
            "value": metric_value
        }
    
    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate if metric is improving, declining, or stable"""
        
        if len(values) < 5:
            return "insufficient_data"
        
        # Compare first 25% vs last 25%
        quarter_len = len(values) // 4
        first_quarter = values.iloc[:quarter_len].mean()
        last_quarter = values.iloc[-quarter_len:].mean()
        
        change = last_quarter - first_quarter
        change_pct = abs(change) / (abs(first_quarter) + 1e-8)
        
        if change_pct < 0.05:
            return "stable"
        elif change > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def _analyze_convergence(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Analyze convergence behavior"""
        
        if 'train_loss' not in df.columns or len(df) < 10:
            return {"status": "insufficient_data"}
        
        train_loss = df['train_loss'].values
        
        # Calculate convergence rate (improvement per epoch)
        first_10 = train_loss[:10].mean()
        last_10 = train_loss[-10:].mean()
        
        improvement = first_10 - last_10
        rate = improvement / len(df)
        
        # Check if still improving
        last_5_improvement = train_loss[-10:-5].mean() - train_loss[-5:].mean()
        still_improving = last_5_improvement > 0.001
        
        # Calculate stability (lower std in recent epochs = more stable)
        stability = 1.0 / (df['train_loss'].tail(10).std() + 1e-8)
        stability_score = min(100, stability * 10)
        
        return {
            "total_improvement": float(improvement),
            "avg_rate_per_epoch": float(rate),
            "still_improving": bool(still_improving),
            "stability_score": float(stability_score),
            "status": "converging" if still_improving else "converged"
        }