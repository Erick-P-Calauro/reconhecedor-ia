"""Camada FaceDetector (ADR 001): deteccao de faces (Spec 002)."""

from .base import FaceDetector
from .yunet_detector import YuNetFaceDetector, ensure_model

__all__ = ["FaceDetector", "YuNetFaceDetector", "ensure_model"]
