"""Interface da camada Model (ADR 001 / Spec 003 RNF-01)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Model(ABC):
    """Camada de rede neural responsavel pela geracao dos embeddings faciais.

    **Interface comum para troca de modelos (Spec 003 RNF-01):** qualquer
    implementacao que cumpra ``embed``/``name`` entra no pipeline sem
    alteracoes nas demais camadas (o pipeline recebe a Model por injecao de
    dependencia).  Uma rede neural futura (FaceNet, ArcFace, ...) substitui a
    baseline apenas implementando esta classe.
    """

    @abstractmethod
    def embed(self, image: np.ndarray) -> np.ndarray:
        """Recebe a face pre-processada e devolve o embedding (vetor 1D)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificacao do modelo, usada no relatorio do experimento (ADR 002)."""

    @property
    def output_dim(self) -> int | None:
        """Dimensao do embedding, quando fixa (informativo)."""
        return None
