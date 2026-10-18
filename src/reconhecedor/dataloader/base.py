"""Interface da camada Dataloader (ADR 001)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Sample


class Dataloader(ABC):
    """Camada responsavel pelo carregamento das imagens dos datasets."""

    @abstractmethod
    def load(self) -> list[Sample]:
        """Carrega todas as amostras do dataset.

        Cada amostra carrega a identidade (pasta de origem) junto, para que
        as camadas seguintes possam treinar/avaliar o reconhecimento.
        """
