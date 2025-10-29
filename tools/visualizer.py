"""
Training Visualizer - Generate charts and plots
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Any, Optional


class TrainingVisualizer:
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def generate(
        self, 
        df: pd.DataFrame, 
        plot_type: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate visualization
        
        Args:
            df: Training log DataFrame
            plot_type: Type of plot (loss_curve, accuracy_curve, learning_rate_schedule, comparison)
            metrics: List of metrics to plot
            
        Returns:
            Dictionary with base64 encoded image and description
        """
        
        if plot_type == "loss_curve":
            return self._plot_loss_curve(df)
        elif plot_type == "accuracy_curve":
            return self._plot_accuracy_curve(df)
        elif plot_type == "learning_rate_schedule":
            return self._plot_learning_rate(df)
        elif plot_type == "comparison":
            return self._plot_comparison(df, metrics)
        else:
            return {"error": f"Unknown plot type: {plot_type}"}
    
    def _plot_loss_curve(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot training and validation loss"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train_loss' in df.columns:
            ax.plot(df['epoch'], df['train_loss'], label='Training Loss', linewidth=2)
        
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            "plot_type": "loss_curve",
            "image_base64": img_base64,
            "description": "Training and validation loss curves over epochs"
        }
    
    def _plot_accuracy_curve(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot training and validation accuracy"""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'train_accuracy' in df.columns:
            ax.plot(df['epoch'], df['train_accuracy'] * 100, label='Training Accuracy', linewidth=2)
        
        if 'val_accuracy' in df.columns:
            ax.plot(df['epoch'], df['val_accuracy'] * 100, label='Validation Accuracy', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            "plot_type": "accuracy_curve",
            "image_base64": img_base64,
            "description": "Training and validation accuracy curves over epochs"
        }
    
    def _plot_learning_rate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Plot learning rate schedule"""
        
        if 'learning_rate' not in df.columns:
            return {"error": "No learning_rate column found"}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['epoch'], df['learning_rate'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            "plot_type": "learning_rate_schedule",
            "image_base64": img_base64,
            "description": "Learning rate schedule over training epochs"
        }
    
    def _plot_comparison(self, df: pd.DataFrame, metrics: Optional[List[str]]) -> Dict[str, Any]:
        """Plot multiple metrics for comparison"""
        
        if metrics is None:
            # Default: plot all available metrics
            metrics = [col for col in ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'] 
                      if col in df.columns]
        
        if not metrics:
            return {"error": "No metrics specified or found"}
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df['epoch'], df[metric], linewidth=2)
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel(metric, fontsize=10)
                ax.set_title(f'{metric} over epochs', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        
        return {
            "plot_type": "comparison",
            "metrics_plotted": metrics,
            "image_base64": img_base64,
            "description": f"Comparison of {', '.join(metrics)} over epochs"
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        return img_base64