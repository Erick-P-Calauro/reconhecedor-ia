"""Detector de face nomeado ``facenet`` (ADR 003) via MTCNN do facenet-pytorch.

O detector que acompanha a FaceNet no ``facenet-pytorch`` e o **MTCNN** (Zhang et
al., 2016). Mantendo o nome literal ``"facenet"`` exigido pelo ADR 003, este
``FaceNetDetector`` usa o MTCNN (pesos embutidos na propria biblioteca, sem
download) para localizar a face mais confiavel da imagem e devolve ``None``
abaixo de ``min_probability`` (Spec 002). O recorte para conter so a face
e herdado de ``FaceDetector.crop``, como no ``YuNetFaceDetector`` — o
detector nao recorta por conta propria. A troca de detector (config.json) segue
a interface comum (Spec 003).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..types import Detection
from .base import FaceDetector

#: Probabilidade minima do MTCNN para aceitar uma deteccao como face.
DEFAULT_MIN_PROBABILITY = 0.9


class FaceNetDetector(FaceDetector):
    """Detector "facenet": MTCNN (facenet-pytorch) para localizar a face.

    Recebe a imagem no formato do pipeline (numpy **BGR**, convencao OpenCV),
    converte para RGB e executa o MTCNN. A instancia do MTCNN e criada sob
    demanda e mantida em cache (nao e reconstruida a cada imagem).
    """

    def __init__(
        self,
        device: str | None = None,
        min_probability: float = DEFAULT_MIN_PROBABILITY,
    ) -> None:
        print("[FaceDetector] inicializando")
        self._device = device
        self.min_probability = float(min_probability)
        self._mtcnn = None  # cache da instancia MTCNN
        print("[FaceDetector] inicializado")

    @property
    def name(self) -> str:
        return "facenet"

    def _get_mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN

            self._mtcnn = MTCNN(keep_all=True, device=self._device)
        return self._mtcnn

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0 or image.ndim != 3:
            return None

        # O pipeline entrega a imagem em BGR (OpenCV); o MTCNN espera RGB.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, probs = self._get_mtcnn().detect(rgb)
        if boxes is None or probs is None or len(boxes) == 0:
            return None

        best = max(
            range(len(boxes)),
            key=lambda index: probs[index] if probs[index] is not None else -1.0,
        )
        score = probs[best]
        if score is None or float(score) < self.min_probability:
            return None

        height, width = image.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in boxes[best][:4])
        x = max(int(np.floor(x1)), 0)
        y = max(int(np.floor(y1)), 0)
        right = min(int(np.ceil(x2)), width)
        bottom = min(int(np.ceil(y2)), height)
        w, h = right - x, bottom - y
        if w <= 0 or h <= 0:
            return None
        return Detection(x=x, y=y, w=w, h=h, score=float(score))
