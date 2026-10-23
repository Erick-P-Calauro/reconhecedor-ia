"""Pre-processamento para modelos PyTorch (CNNs) da camada Model (ADR 001).

Deixa o recorte da face no formato esperado por redes convolucionais: RGB,
tamanho fixo, escala [0, 1] e normalizacao por media/desvio, devolvendo um
tensor CHW em ``float32``. E uma implementacao **generica** (nomes neutros),
reutilizavel por qualquer experimento cujo ``face_recognition_model`` seja uma
CNN PyTorch/torchvision (ADR 002).
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import PreProcessor

#: Preset ImageNet (torchvision).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Preset FaceNet: ``fixed_image_standardization`` aproximada para [-1, 1].
FACENET_MEAN = (0.5, 0.5, 0.5)
FACENET_STD = (0.5, 0.5, 0.5)


class TorchPreProcessor(PreProcessor):
    """Prepara o recorte da face para uma CNN PyTorch (Spec 002 RF-03).

    - converte BGR -> RGB (convencao torchvision);
    - redimensiona para ``size`` (largura, altura);
    - normaliza ``(pixel/255 - mean) / std`` por canal;
    - devolve um array ``float32`` com layout CHW.
    """

    def __init__(
        self,
        size: tuple[int, int] = (224, 224),
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        name: str = "torch",
    ) -> None:
        print("[PreProcessor] inicializando")
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("size deve ter dimensoes positivas")
        self.size = (int(size[0]), int(size[1]))  # (largura, altura)
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self._name = name
        print("[PreProcessor] inicializado")

    @property
    def name(self) -> str:
        return f"{self._name}-{self.size[0]}x{self.size[1]}"

    def process(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("imagem vazia recebida pelo PreProcessor")
        if image.ndim == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        interpolation = cv2.INTER_AREA if rgb.shape[1] > self.size[0] else cv2.INTER_LINEAR
        resized = cv2.resize(rgb, self.size, interpolation=interpolation)
        chw = np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))
        return ((chw - self.mean) / self.std).astype(np.float32)
