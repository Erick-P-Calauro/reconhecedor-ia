"""Interface da camada PreProcessor (ADR 001)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PreProcessor(ABC):
    """Camada que processa imagens para o formato adequado a camada Model."""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Recebe o recorte da face e devolve a entrada padronizada da Model."""
