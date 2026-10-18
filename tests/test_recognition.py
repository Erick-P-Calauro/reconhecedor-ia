"""Testes de Model, PreProcessor e FaceRecognition (Spec 003)."""

from __future__ import annotations

import numpy as np
import pytest

from reconhecedor.face_recognition import FaceRecognition
from reconhecedor.model import Model
from reconhecedor.preprocessor import StandardPreProcessor
from support import StubEmbeddingModel


def _face_like(size: int = 128, shift: int = 0, seed: int = 0) -> np.ndarray:
    """Imagem sintetica 'cara-like' (rotulos claros/escuros) para testes."""
    rng = np.random.default_rng(seed)
    base = np.linspace(30, 220, size, dtype=np.float32)
    image = np.tile(base, (size, 1))
    if shift:
        image = np.roll(image, shift, axis=1)
    image = np.clip(image + rng.normal(0, 3, image.shape), 0, 255)
    return image.astype(np.uint8)


def test_preprocessor_output_is_fixed_grayscale_normalized():
    preprocessor = StandardPreProcessor(size=(64, 64))
    output = preprocessor.process(_face_like())
    assert output.shape == (64, 64)
    assert output.dtype == np.float32
    assert 0.0 <= output.min() and output.max() <= 1.0
    with pytest.raises(ValueError):
        preprocessor.process(np.zeros((0, 0), dtype=np.uint8))


def test_model_is_deterministic_and_l2_normalized():
    model = StubEmbeddingModel()
    assert isinstance(model, Model)  # interface comum (Spec 003 RNF-01)
    preprocessor = StandardPreProcessor(size=(128, 128))
    vector = model.embed(preprocessor.process(_face_like()))
    again = model.embed(preprocessor.process(_face_like()))
    assert vector.shape == (model.output_dim,)
    assert np.allclose(vector, again)  # deterministico
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def test_model_embeds_black_image_without_error():
    model = StubEmbeddingModel()
    vector = model.embed(np.zeros((128, 128), dtype=np.float32))
    assert np.isfinite(vector).all()


def test_recognizes_enrolled_person():
    """Spec 003 RF-03: pessoa cadastrada e reconhecida."""
    recognition = FaceRecognition(threshold=0.5)
    recognition.enroll("alice", np.array([1.0, 0.0, 0.0]))
    recognition.enroll("bob", np.array([0.0, 1.0, 0.0]))

    result = recognition.identify(np.array([0.98, 0.17, 0.0]))

    assert result.predicted_label == "alice"
    assert result.similarity >= 0.5


def test_classifies_unknown_person():
    """Spec 003 RF-04: fora do dominio -> desconhecido (predicted None)."""
    recognition = FaceRecognition(threshold=0.5)
    recognition.enroll("alice", np.array([1.0, 0.0, 0.0]))

    result = recognition.identify(np.array([0.0, 0.0, 1.0]))

    assert result.predicted_label is None
    assert result.best_label == "alice"


def test_empty_gallery_identifies_everyone_as_unknown():
    recognition = FaceRecognition(threshold=0.5)
    assert recognition.identify(np.array([1.0, 0.0])).predicted_label is None


def test_threshold_validation():
    with pytest.raises(ValueError):
        FaceRecognition(threshold=1.5)


def test_threshold_has_no_default_in_base_layer():
    """ADR 002: a base nao embute limiar calibrado de experimento (ex.: 0.8)."""
    with pytest.raises(TypeError):
        FaceRecognition()
