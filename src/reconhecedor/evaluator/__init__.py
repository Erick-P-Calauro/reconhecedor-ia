"""Camada Evaluator (ADR 001): metricas e avaliacao de experimentos (Spec 004)."""

from .evaluator import Evaluator
from .metrics import Metrics, compute_metrics
from .threshold_optimizer import ThresholdOptimizer, ThresholdOptimizationResult
from .plots import generate_plots

__all__ = ["Evaluator", "Metrics", "compute_metrics", "ThresholdOptimizer", "ThresholdOptimizationResult", "generate_plots"]
