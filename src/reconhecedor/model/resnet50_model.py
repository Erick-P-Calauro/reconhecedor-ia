"""Model ResNet50 (torchvision) como extrator de embeddings (ADR 001 / ADR 003).

Implementacao **generica** da camada Model: usa a ResNet50 pre-treinada em
ImageNet (pesos ``torchvision``) sem a camada de classificacao. O vetor de 2048
dimensoes do pooling global e usado como embedding facial. Reutilizavel por
qualquer experimento cujo ``face_recognition_model`` seja ``resnet50`` (ADR 002,
ADR 003), sem alterar o pipeline (Spec 003).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .base import Model
from .torch_utils import to_batch

#: Pesos ImageNet usados por padrao.
DEFAULT_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


class ResNet50Model(Model):
    """ResNet50 (ImageNet) como extrator de caracteristicas faciais."""

    def __init__(self, weights: ResNet50_Weights = DEFAULT_WEIGHTS, device: str | None = None) -> None:
        print("[Model] inicializando")
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        network = resnet50(weights=weights, progress=False)
        network.fc = nn.Identity()  # descarta a classificacao: mantem as features
        network.eval()
        self._network = network.to(self._device)
        print("[Model] inicializado")

    @property
    def name(self) -> str:
        return "resnet50"

    @property
    def output_dim(self) -> int:
        return 2048

    def embed(self, image: np.ndarray) -> np.ndarray:
        batch = to_batch(image, self._device)
        with torch.no_grad():
            features = self._network(batch)
        return features.reshape(-1).cpu().numpy().astype(np.float32)
