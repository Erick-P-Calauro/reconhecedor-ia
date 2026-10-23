"""Testes da camada FaceDetector (Spec 002)."""

from __future__ import annotations

import numpy as np
import pytest

from reconhecedor.face_detector import YuNetFaceDetector
from reconhecedor.types import Detection


@pytest.fixture(scope="module")
def detector() -> YuNetFaceDetector:
    return YuNetFaceDetector()


def _noise_image(seed: int = 42, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_detects_no_face_in_noise(detector: YuNetFaceDetector):
    """Imagem sem face -> detector sinaliza ausencia (retorna None)."""
    assert detector.detect(_noise_image()) is None


def test_detects_no_face_in_blank_image(detector: YuNetFaceDetector):
    assert detector.detect(np.full((256, 256, 3), 255, dtype=np.uint8)) is None


def test_detects_no_face_in_empty_image(detector: YuNetFaceDetector):
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_crop_returns_none_without_detection(detector: YuNetFaceDetector):
    """Pipeline usa None para descartar; crop nunca e chamado."""
    image = _noise_image()
    assert detector.detect(image) is None
    assert detector.crop(image, None) is None


def test_crop_returns_only_face_region(detector: YuNetFaceDetector):
    """Recorte contem a face com margem e nao excede a imagem."""
    image = np.full((300, 400, 3), 128, dtype=np.uint8)
    detection = Detection(x=100, y=80, w=80, h=100, score=0.99)
    crop = detector.crop(image, detection, padding=0.25)
    assert crop is not None
    assert crop.shape[0] < image.shape[0] and crop.shape[1] < image.shape[1]
    # Margem: x=[80, 200], y=[55, 205].
    assert crop.shape[1] == 120 and crop.shape[0] == 150


def test_crop_clamps_to_image_borders(detector: YuNetFaceDetector):
    image = np.full((50, 50, 3), 0, dtype=np.uint8)
    detection = Detection(x=0, y=0, w=60, h=60, score=0.9)
    crop = detector.crop(image, detection, padding=0.5)
    assert crop is not None
    assert crop.shape[:2] == (50, 50)
