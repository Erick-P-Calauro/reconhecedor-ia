"""Camada FaceDetector (ADR 001): deteccao de faces (Spec 002)."""

from .base import FaceDetector
from .facenet_detector import FaceNetDetector
from .yunet_detector import YuNetFaceDetector

__all__ = ["FaceDetector", "YuNetFaceDetector", "FaceNetDetector"]
