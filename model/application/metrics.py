# metrics.py - Monitoring & Metrics
# ============================================================================

import time
from dataclasses import dataclass, field
from typing import Dict, List
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class AuditMetrics:
    """Audit performance metrics."""
    total_criteria: int = 0
    total_time: float = 0.0
    time_per_criterion: Dict[str, float] = field(default_factory=dict)
    attempts_per_criterion: Dict[str, int] = field(default_factory=dict)
    confidence_per_criterion: Dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    
    def add_criterion(self, criterion: str, time_spent: float, attempts: int, confidence: float):
        """Adds metrics for a criterion."""
        self.total_criteria += 1
        self.time_per_criterion[criterion] = time_spent
        self.attempts_per_criterion[criterion] = attempts
        self.confidence_per_criterion[criterion] = confidence
    
    def display_summary(self):
        """Displays metrics summary."""
        table = Table(title="📊 Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        avg_time = sum(self.time_per_criterion.values()) / len(self.time_per_criterion) if self.time_per_criterion else 0
        avg_attempts = sum(self.attempts_per_criterion.values()) / len(self.attempts_per_criterion) if self.attempts_per_criterion else 0
        avg_confidence = sum(self.confidence_per_criterion.values()) / len(self.confidence_per_criterion) if self.confidence_per_criterion else 0
        
        table.add_row("Total Criteria", str(self.total_criteria))
        table.add_row("Total Time", f"{self.total_time:.2f}s")
        table.add_row("Avg Time/Criterion", f"{avg_time:.2f}s")
        table.add_row("Avg Attempts", f"{avg_attempts:.1f}")
        table.add_row("Avg Confidence", f"{avg_confidence:.0%}")
        
        console.print(table)


class MetricsTracker:
    """Metrics tracker during the audit."""
    
    def __init__(self):
        self.metrics = AuditMetrics()
        self.start_time = None
        self.criterion_start_time = None
    
    def start_audit(self):
        """Starts audit tracking."""
        self.start_time = time.time()
    
    def finish_audit(self):
        """Finishes audit tracking."""
        if self.start_time:
            self.metrics.total_time = time.time() - self.start_time
    
    def start_criterion(self):
        """Starts criterion tracking."""
        self.criterion_start_time = time.time()
    
    def finish_criterion(self, criterion: str, attempts: int, confidence: float):
        """Finishes criterion tracking."""
        if self.criterion_start_time:
            time_spent = time.time() - self.criterion_start_time
            self.metrics.add_criterion(criterion, time_spent, attempts, confidence)
    
    def get_metrics(self) -> AuditMetrics:
        """Returns collected metrics."""
        return self.metrics