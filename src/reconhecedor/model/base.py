"""Interface da camada Model (ADR 001 / Spec 003)."""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):
    """Camada de rede neural responsavel pela geracao dos embeddings faciais."""

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

    def fine_tune(self, train_samples, *, detector, preprocessor, crop_padding, epochs=5, lr=1e-4) -> None:
        """Fine-tuning (default no-op para modelos nao ajustaveis)."""
        return None

    @property
    def is_fine_tunable(self) -> bool:
        """Indica se o modelo suporta fine-tuning."""
        return False