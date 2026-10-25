"""Experimento ADR 003 — Deteccao YuNet + Reconhecimento Facenet (dataset LFW)."""

from __future__ import annotations

from pathlib import Path

from experiments import layers
from experiments.config import ExperimentConfig, load_config
from reconhecedor.pipeline import ExperimentResult

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load() -> ExperimentConfig:
    """Carrega e valida a configuracao replicavel do experimento."""
    return load_config(CONFIG_PATH)


def build_layers(config: ExperimentConfig, *, detector_model_path=None, rematerialize=False) -> dict:
    """Instancia as seis camadas da ADR 001 a partir da configuracao."""
    return layers.build_layers(config, detector_model_path=detector_model_path, rematerialize=rematerialize)


def run(
    *,
    dataset=None,
    output_dir=None,
    threshold=None,
    crop_padding=None,
    image_size=None,
    detector_model_path=None,
    rematerialize=False,
    sample_size=None,
    n_classes=None,
    min_photos_per_class=None,
    validation_ratio=None,
    test_ratio=None,
    train_ratio=None,
    fine_tune_epochs=None,
    learning_rate=None,
    seed=None,
) -> ExperimentResult:
    """Executa o experimento ponta a ponta, gravando em ``config.results_dir``."""
    return layers.run(
        CONFIG_PATH,
        dataset=dataset,
        output_dir=output_dir,
        threshold=threshold,
        crop_padding=crop_padding,
        image_size=image_size,
        detector_model_path=detector_model_path,
        rematerialize=rematerialize,
        sample_size=sample_size,
        n_classes=n_classes,
        min_photos_per_class=min_photos_per_class,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        train_ratio=train_ratio,
        fine_tune_epochs=fine_tune_epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
