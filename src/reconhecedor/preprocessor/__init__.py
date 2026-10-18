"""Camada PreProcessor (ADR 001): prepara imagens para a camada Model."""

from .base import PreProcessor
from .standard import StandardPreProcessor

__all__ = ["PreProcessor", "StandardPreProcessor"]
