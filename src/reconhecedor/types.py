"""Tipos compartilhados entre as camadas do pipeline (ADR 001)."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True, eq=False)
class Sample:
    """Imagem carregada pelo Dataloader, com a identidade da pasta de origem."""

    path: Path
    image: np.ndarray  # BGR (convenção OpenCV)
    label: str  # identidade = nome da pasta no dataset


@dataclass(frozen=True)
class Detection:
    """Face detectada: caixa (x, y, largura, altura) em pixels e confianca."""

    x: int
    y: int
    w: int
    h: int
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    """Resultado da comparacao de embeddings pela camada FaceRecognition.

    ``predicted_label`` e ``None`` quando a pessoa e desconhecida
    (fora do dominio cadastrado - Spec 003).
    """

    predicted_label: str | None
    similarity: float
    best_label: str | None


@dataclass(frozen=True, eq=False)
class PredictionRecord:
    """Predicao de uma imagem de probe, usada pelo Evaluator (Spec 004)."""

    path: Path
    true_label: str
    predicted_label: str | None
    similarity: float
