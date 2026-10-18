"""Camada PreProcessor (ADR 001): prepara imagens para a camada Model."""

from .base import PreProcessor
from .standard import StandardPreProcessor
from .torch_preprocessor import TorchPreProcessor

__all__ = ["PreProcessor", "StandardPreProcessor", "TorchPreProcessor"]
