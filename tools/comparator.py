"""
Epoch Comparator - Compare metrics across epochs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class EpochComparator:
    def compare(
        self, 
        df: pd.DataFrame, 
        epoch_range: Optional[List[int]] = None,
        comparison_type: str = "sequential"
    ) -> Dict[str, Any]:
        """
        Compare epochs or epoch ranges
        
        Args:
            df: Training log DataFrame
            epoch_range: [start_epoch, end_epoch] or None for auto
            comparison_type: sequential, interval, or best_vs_worst
            
        Returns:
            Comparison results
        """
        
        if comparison_type == "sequential":
            return self._compare_sequential(df, epoch_range)
        elif comparison_type == "interval":
            return self._compare_interval(df, epoch_range)
        elif comparison_type == "best_vs_worst":
            return self._compare_best_vs_worst(df)
        else:
            return {"error": f"Unknown comparison type: {comparison_type}"}
    
    def _compare_sequential(self, df: pd.DataFrame, epoch_range: Optional[List[int]]) -> Dict[str, Any]:
        """Compare consecutive epochs"""
        
        if epoch_range:
            start, end = epoch_range
            df_subset = df[(df['epoch'] >= start) & (df['epoch'] <= end)]
        else:
            df_subset = df
        
        # Calculate changes between consecutive epochs
        changes = {}
        numeric_cols = df_subset.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'epoch':
                continue
            
            values = df_subset[col].values
            diffs = np.diff(values)
            
            changes[col] = {
                "avg_change": float(np.mean(diffs)),
                "max_increase": float(np.max(diffs)),
                "max_decrease": float(np.min(diffs)),
                "volatility": float(np.std(diffs))
            }
        
        return {
            "comparison_type": "sequential",
            "epoch_range": [int(df_subset['epoch'].min()), int(df_subset['epoch'].max())],
            "epochs_analyzed": len(df_subset) - 1,
            "changes": changes
        }
    
    def _compare_interval(self, df: pd.DataFrame, epoch_range: Optional[List[int]]) -> Dict[str, Any]:
        """Compare two intervals of epochs"""
        
        if not epoch_range or len(epoch_range) != 2:
            # Default: compare first 25% vs last 25%
            total = len(df)
            quarter = total // 4
            interval1 = df.iloc[:quarter]
            interval2 = df.iloc[-quarter:]
            range_desc = f"First {quarter} vs Last {quarter} epochs"
        else:
            mid = (epoch_range[0] + epoch_range[1]) // 2
            interval1 = df[(df['epoch'] >= epoch_range[0]) & (df['epoch'] < mid)]
            interval2 = df[(df['epoch'] >= mid) & (df['epoch'] <= epoch_range[1])]
            range_desc = f"Epochs {epoch_range[0]}-{mid} vs {mid}-{epoch_range[1]}"
        
        # Compare metrics between intervals
        comparisons = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'epoch':
                continue
            
            mean1 = interval1[col].mean()
            mean2 = interval2[col].mean()
            
            change = mean2 - mean1
            change_pct = (change / abs(mean1)) * 100 if mean1 != 0 else 0
            
            comparisons[col] = {
                "interval1_mean": float(mean1),
                "interval2_mean": float(mean2),
                "absolute_change": float(change),
                "percent_change": float(change_pct),
                "improved": bool(
                    (change > 0 and 'acc' in col.lower()) or 
                    (change < 0 and 'loss' in col.lower())
                )
            }
        
        return {
            "comparison_type": "interval",
            "description": range_desc,
            "interval1_epochs": int(len(interval1)),
            "interval2_epochs": int(len(interval2)),
            "comparisons": comparisons
        }
    
    def _compare_best_vs_worst(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare best performing epoch vs worst"""
        
        # Find best and worst based on validation accuracy or loss
        if 'val_accuracy' in df.columns:
            best_idx = df['val_accuracy'].idxmax()
            worst_idx = df['val_accuracy'].idxmin()
            metric = 'val_accuracy'
        elif 'val_loss' in df.columns:
            best_idx = df['val_loss'].idxmin()
            worst_idx = df['val_loss'].idxmax()
            metric = 'val_loss'
        else:
            return {"error": "No validation metrics found"}
        
        best_epoch = df.loc[best_idx]
        worst_epoch = df.loc[worst_idx]
        
        # Compare all metrics
        comparisons = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col == 'epoch':
                continue
            
            best_val = float(best_epoch[col])
            worst_val = float(worst_epoch[col])
            diff = best_val - worst_val
            
            comparisons[col] = {
                "best_epoch_value": best_val,
                "worst_epoch_value": worst_val,
                "difference": diff
            }
        
        return {
            "comparison_type": "best_vs_worst",
            "best_epoch": int(best_epoch['epoch']),
            "worst_epoch": int(worst_epoch['epoch']),
            "comparison_metric": metric,
            "comparisons": comparisons
        }