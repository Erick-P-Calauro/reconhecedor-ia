"""Camada Model (ADR 001): geracao dos embeddings faciais."""

from .base import Model
from .baseline_hog import BaselineHogModel

__all__ = ["Model", "BaselineHogModel"]
