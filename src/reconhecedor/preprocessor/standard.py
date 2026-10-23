"""Pre-processamento padrao: escala de cinza, tamanho fixo e normalizacao."""

from __future__ import annotations

import cv2
import numpy as np

from .base import PreProcessor


class StandardPreProcessor(PreProcessor):
    """Converte para cinza, redimensiona para ``size`` e normaliza para [0, 1].

    A saida e sempre um ``float32`` 2D de dimensoes fixas, garantindo que a
    camada Model receba entradas homogeneas independente da imagem original.
    """

    def __init__(self, size: tuple[int, int] = (128, 128)) -> None:
        print("[PreProcessor] inicializando")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("size deve ter dimensoes positivas")
        self.size = (int(size[0]), int(size[1]))  # (largura, altura)
        print("[PreProcessor] inicializado")

    @property
    def name(self) -> str:
        return f"standard-gray-{self.size[0]}x{self.size[1]}"

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("imagem vazia recebida pelo PreProcessor")
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        interpolation = cv2.INTER_AREA if gray.shape[1] > self.size[0] else cv2.INTER_LINEAR
        resized = cv2.resize(gray, self.size, interpolation=interpolation)
        return resized.astype(np.float32) / 255.0
