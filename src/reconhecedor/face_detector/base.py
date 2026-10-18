"""Interface da camada FaceDetector (ADR 001 / Spec 002)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import Detection


class FaceDetector(ABC):
    """Camada responsavel pela funcionalidade de deteccao de faces.

    - RF-01: verificar se uma imagem contem faces.
    - RF-02: imagens sem face sao descartadas pelo pipeline (retornar ``None``).
    - RF-03: ``crop`` recorta a imagem para conter apenas a face.
    """

    @abstractmethod
    def detect(self, image: np.ndarray) -> Detection | None:
        """Retorna a face mais confiavel da imagem ou ``None`` se nao houver."""

    def crop(
        self,
        image: np.ndarray,
        detection: Detection | None,
        padding: float = 0.25,
    ) -> np.ndarray | None:
        """RF-03: recorta a regiao da face (com margem de contexto) da imagem.

        ``padding`` e a fracao da largura/altura da caixa adicionada em cada
        lado, para nao cortar a face tao rente.
        """
        if detection is None:
            return None
        height, width = image.shape[:2]
        pad_x = int(detection.w * padding)
        pad_y = int(detection.h * padding)
        x0 = max(detection.x - pad_x, 0)
        y0 = max(detection.y - pad_y, 0)
        x1 = min(detection.x + detection.w + pad_x, width)
        y1 = min(detection.y + detection.h + pad_y, height)
        if x1 <= x0 or y1 <= y0:
            return None
        return image[y0:y1, x0:x1].copy()
