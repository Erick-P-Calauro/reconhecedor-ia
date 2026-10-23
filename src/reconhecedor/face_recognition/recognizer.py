"""Comparacao de embeddings e reconhecimento de identidades (Spec 003)."""

from __future__ import annotations
import numpy as np
from ..types import RecognitionResult

class FaceRecognition:
    """Camada responsavel por comparar os embeddings faciais.

    - RF-03: pessoas previamente inseridas no dominio sao reconhecidas
      (``enroll`` + maior similaridade de cosseno >= limiar).
    - RF-04: pessoas fora do dominio sao classificadas como desconhecidas
      (similaridade abaixo do limiar => ``predicted_label = None``).

    ``threshold`` e **obrigatorio**: a pasta base nao calibra limiar de
    experimento (ADR 002) — o valor vem da configuracao do experimento
    (``src/experiments/<nome>/config.json``).
    """

    def __init__(self, threshold: float) -> None:
        print("[FaceRecognition] inicializando")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold deve estar em [0, 1]")
        self.threshold = float(threshold)
        self._gallery: dict[str, list[np.ndarray]] = {}
        print("[FaceRecognition] inicializado")

    @property
    def name(self) -> str:
        return f"cosine-threshold-{self.threshold:.2f}"

    def enroll(self, label: str, embedding: np.ndarray) -> None:
        """Cadastra um embedding da identidade ``label`` no dominio."""
        if not label:
            raise ValueError("label vazio nao pode ser cadastrado")
        self._gallery.setdefault(label, []).append(self._normalize(embedding))

    def identify(self, embedding: np.ndarray) -> RecognitionResult:
        """Compara o embedding com o dominio e devolve a decisao de identidade. Retorna ``predicted_label = None`` para desconhecidos (RF-04)."""
        if not self._gallery:
            return RecognitionResult(predicted_label=None, similarity=0.0, best_label=None)

        vector = self._normalize(embedding)
        best_label: str | None = None
        best_similarity = -1.0
        for label, embeddings in self._gallery.items():
            centroid = self._normalize(np.mean(embeddings, axis=0))
            similarity = float(np.dot(vector, centroid))
            if similarity > best_similarity:
                best_label, best_similarity = label, similarity

        predicted = best_label if best_similarity >= self.threshold else None
        return RecognitionResult(
            predicted_label=predicted,
            similarity=best_similarity,
            best_label=best_label,
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(array))
        if norm > 0:
            array = array / norm
        return array
