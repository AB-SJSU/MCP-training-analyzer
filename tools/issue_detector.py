"""
Training Issue Detector - Detects overfitting, divergence, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class IssueDetector:
    def detect_all(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all issue detection checks
        
        Args:
            df: Training log DataFrame
            
        Returns:
            Dictionary with detected issues
        """
        issues = []
        
        # Check for overfitting
        overfitting = self.detect_overfitting(df)
        if overfitting['detected']:
            issues.append(overfitting)
        
        # Check for divergence
        divergence = self.detect_divergence(df)
        if divergence['detected']:
            issues.append(divergence)
        
        # Check for plateau
        plateau = self.detect_plateau(df)
        if plateau['detected']:
            issues.append(plateau)
        
        # Check for slow convergence
        slow_conv = self.detect_slow_convergence(df)
        if slow_conv['detected']:
            issues.append(slow_conv)
        
        # Calculate health score
        health_score = self._calculate_health_score(issues)
        
        return {
            "issues_detected": issues,
            "total_issues": len(issues),
            "health_score": health_score,
            "summary": self._generate_summary(issues)
        }
    
    def detect_overfitting(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overfitting by comparing train vs validation metrics"""
        
        # Check if we have required columns
        if 'train_accuracy' not in df.columns or 'val_accuracy' not in df.columns:
            return {"detected": False}
        
        # Get final 10 epochs
        recent_epochs = df.tail(10)
        
        avg_train_acc = recent_epochs['train_accuracy'].mean()
        avg_val_acc = recent_epochs['val_accuracy'].mean()
        
        gap = avg_train_acc - avg_val_acc
        
        # Overfitting if gap > 15%
        if gap > 0.15:
            severity = "high" if gap > 0.25 else "medium"
            
            return {
                "detected": True,
                "type": "overfitting",
                "severity": severity,
                "train_accuracy": float(avg_train_acc),
                "val_accuracy": float(avg_val_acc),
                "gap": float(gap),
                "description": f"Training accuracy ({avg_train_acc:.1%}) significantly higher than validation ({avg_val_acc:.1%}). Gap of {gap:.1%} indicates overfitting.",
                "detected_at_epoch": int(df['epoch'].iloc[-10])
            }
        
        return {"detected": False}
    
    def detect_divergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect training divergence (loss explosion)"""
        
        if 'train_loss' not in df.columns:
            return {"detected": False}
        
        losses = df['train_loss'].values
        
        # Check for sudden spikes (loss doubles or more)
        for i in range(1, len(losses)):
            if losses[i] > losses[i-1] * 2 and losses[i] > 2.0:
                return {
                    "detected": True,
                    "type": "divergence",
                    "severity": "critical",
                    "epoch": int(df['epoch'].iloc[i]),
                    "loss_before": float(losses[i-1]),
                    "loss_after": float(losses[i]),
                    "description": f"Training diverged at epoch {df['epoch'].iloc[i]}. Loss jumped from {losses[i-1]:.3f} to {losses[i]:.3f}."
                }
        
        return {"detected": False}
    
    def detect_plateau(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect validation accuracy plateau"""
        
        if 'val_accuracy' not in df.columns or len(df) < 20:
            return {"detected": False}
        
        # Check last 20 epochs
        recent = df.tail(20)
        val_acc = recent['val_accuracy'].values
        
        # Calculate variance
        variance = np.var(val_acc)
        
        # Plateau if variance very low (< 0.0001) and not at high accuracy
        if variance < 0.0001 and val_acc.mean() < 0.95:
            return {
                "detected": True,
                "type": "plateau",
                "severity": "medium",
                "plateaued_at": float(val_acc.mean()),
                "variance": float(variance),
                "description": f"Validation accuracy plateaued at {val_acc.mean():.1%} for last 20 epochs. Model stopped learning.",
                "detected_at_epoch": int(df['epoch'].iloc[-20])
            }
        
        return {"detected": False}
    
    def detect_slow_convergence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect slow learning"""
        
        if 'train_accuracy' not in df.columns or len(df) < 30:
            return {"detected": False}
        
        # Check improvement over first 30 epochs
        initial_acc = df['train_accuracy'].iloc[:5].mean()
        epoch_30_acc = df['train_accuracy'].iloc[25:30].mean()
        
        improvement = epoch_30_acc - initial_acc
        
        # Slow if improved less than 15% in 30 epochs
        if improvement < 0.15:
            return {
                "detected": True,
                "type": "slow_convergence",
                "severity": "medium",
                "initial_accuracy": float(initial_acc),
                "epoch_30_accuracy": float(epoch_30_acc),
                "improvement": float(improvement),
                "description": f"Training progressing slowly. Only {improvement:.1%} improvement in first 30 epochs. Consider increasing learning rate."
            }
        
        return {"detected": False}
    
    def _calculate_health_score(self, issues: List[Dict]) -> int:
        """Calculate overall health score (0-100)"""
        
        if not issues:
            return 100
        
        # Deduct points based on severity
        score = 100
        
        for issue in issues:
            severity = issue.get('severity', 'low')
            if severity == 'critical':
                score -= 40
            elif severity == 'high':
                score -= 25
            elif severity == 'medium':
                score -= 15
            else:
                score -= 5
        
        return max(0, score)
    
    def _generate_summary(self, issues: List[Dict]) -> str:
        """Generate human-readable summary"""
        
        if not issues:
            return "No major issues detected. Training looks healthy!"
        
        issue_types = [issue['type'] for issue in issues]
        
        summary = f"Found {len(issues)} issue(s): {', '.join(issue_types)}."
        
        # Add specific advice
        if 'overfitting' in issue_types:
            summary += " Consider adding regularization (dropout, weight decay) or early stopping."
        if 'divergence' in issue_types:
            summary += " Training diverged - reduce learning rate significantly or use gradient clipping."
        if 'plateau' in issue_types:
            summary += " Learning plateaued - try adjusting learning rate or architecture."
        if 'slow_convergence' in issue_types:
            summary += " Learning too slow - consider increasing learning rate."
        
        return summary