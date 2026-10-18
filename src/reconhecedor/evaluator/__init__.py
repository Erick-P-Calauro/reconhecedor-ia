"""Camada Evaluator (ADR 001): metricas e avaliacao de experimentos (Spec 004)."""

from .evaluator import Evaluator
from .metrics import Metrics, compute_metrics

__all__ = ["Evaluator", "Metrics", "compute_metrics"]
