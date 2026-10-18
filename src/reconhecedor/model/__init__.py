"""Camada Model (ADR 001): geracao dos embeddings faciais.

Apenas a **interface** vive na pasta base; implementacoes concretas usadas por
experimentos (ex.: ``BaselineHogModel``) ficam em ``src/experiments/<nome>/``
(ADR 002 — nenhuma implementacao de experimento na pasta base das camadas).
"""

from .base import Model

__all__ = ["Model"]
