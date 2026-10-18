"""Detector de face YuNet (OpenCV ``FaceDetectorYN``), leve e sem DL pesada."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from ..types import Detection
from .base import FaceDetector

MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

#: ``src/reconhecedor/face_detector/`` -> raiz do projeto.
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[3] / "models" / "face_detection_yunet_2023mar.onnx"
)


def ensure_model(model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Garante o modelo publico do detector; baixa automaticamente se ausente."""
    model_path = Path(model_path)
    if model_path.is_file() and model_path.stat().st_size > 10_000:
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except OSError as exc:
        raise RuntimeError(
            f"Nao foi possivel baixar o modelo do detector ({MODEL_URL}). "
            "Baixe manualmente e indique o caminho com --detector-model."
        ) from exc
    if not model_path.is_file() or model_path.stat().st_size < 10_000:
        raise RuntimeError("Download do modelo do detector falhou (arquivo invalido).")
    return model_path


class YuNetFaceDetector(FaceDetector):
    """FaceDetector baseado no YuNet (Spec 002).

    Leve (ONNX ~230 KB executado pelo proprio OpenCV), com limiar de
    confianca configuravel. Retorna ``None`` quando nao ha face (RF-01),
    permitindo ao pipeline descartar a imagem (RF-02).
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = ensure_model(model_path)
        self.score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._detector: cv2.FaceDetectorYN | None = None

    @property
    def name(self) -> str:
        return "yunet"

    def _get(self) -> "cv2.FaceDetectorYN":
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN_create(
                str(self.model_path),
                "",
                (320, 320),
                self.score_threshold,
                self._nms_threshold,
                self._top_k,
            )
        return self._detector

    def detect(self, image: np.ndarray) -> Detection | None:
        if image is None or image.size == 0 or image.ndim != 3:
            return None
        height, width = image.shape[:2]
        detector = self._get()
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda face: face[-1])
        x, y, w, h = (int(value) for value in best[:4])
        if w <= 0 or h <= 0:
            return None
        return Detection(x=x, y=y, w=w, h=h, score=float(best[-1]))
