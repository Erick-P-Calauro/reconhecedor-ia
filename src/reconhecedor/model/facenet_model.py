"""Model FaceNet (Inception-ResNet-v1) via ``facenet-pytorch`` (ADR 003).

Implementacao **generica** da camada Model: usa a biblioteca ``facenet-pytorch``
para carregar a FaceNet (Schroff et al., 2015) pre-treinada no **VGGFace2** com
``InceptionResnetV1(pretrained="vggface2", classify=False)``. A propria biblioteca
baixa/gerencia o checkpoint no cache do torch (``~/.cache/torch/checkpoints``).
Reutilizavel por qualquer experimento cujo ``face_recognition_model`` seja
``facenet`` (ADR 002, ADR 003); a troca de modelo nao exige mudar o pipeline
(Spec 003).

A saida e um embedding de 512 dimensoes com norma L2 unitaria. A entrada esperada
e a saida do ``TorchPreProcessor`` preset FaceNet: CHW ``float32`` em ``[-1, 1]``
(``(pixel/255 - 0.5) / 0.5``), equivalente ao ``fixed_image_standardization``
``(x - 127.5) / 128`` usado pelo ``InceptionResnetV1`` (diferenca apenas no fator
de escala 127,5 vs. 128, ~0,4%).
"""

from __future__ import annotations

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F

from .base import Model
from .torch_utils import to_batch


class FaceNetModel(Model):
    """FaceNet (Inception-ResNet-v1, VGGFace2) como extrator de embeddings.

    Carrega a rede pre-treinada via ``facenet-pytorch`` e a mantem em modo
    avaliacao; o weight do checkpoint fica no cache do torch, gerenciado pela
    biblioteca.
    """

    def __init__(self, device: str | None = None) -> None:
        print("[Model] inicializando")
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = (
            InceptionResnetV1(pretrained="vggface2", classify=False).eval().to(self._device)
        )
        print("[Model] inicializado")

    @property
    def name(self) -> str:
        return "facenet"

    @property
    def output_dim(self) -> int:
        return 512

    @property
    def is_fine_tunable(self) -> bool:
        return False

    def embed(self, image: np.ndarray) -> np.ndarray:
        batch = to_batch(image, self._device)
        with torch.no_grad():
            embedding = self._network(batch)
        # O forward(classify=False) do facenet-pytorch ja aplica L2; renormalizamos
        # para garantir norma unitaria independentemente da versao da biblioteca.
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.reshape(-1).cpu().numpy().astype(np.float32)
