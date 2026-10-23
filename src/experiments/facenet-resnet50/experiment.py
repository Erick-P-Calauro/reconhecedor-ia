"""Experimento ADR 003 — Deteccao FaceNet + Reconhecimento ResNet50 (dataset LFW).

Instancia as seis camadas da ADR 001 a partir de ``config.json`` (configuracao
replicavel) e executa o pipeline; os resultados ficam em ``results/`` (ADR 002).
"""

from __future__ import annotations

from pathlib import Path

from experiments import layers
from experiments.config import ExperimentConfig, load_config
from reconhecedor.pipeline import ExperimentResult

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load() -> ExperimentConfig:
    """Carrega e valida a configuracao replicavel do experimento."""
    return load_config(CONFIG_PATH)


def build_layers(config: ExperimentConfig, *, detector_model_path=None) -> dict:
    """Instancia as seis camadas da ADR 001 a partir da configuracao."""
    return layers.build_layers(config, detector_model_path=detector_model_path)


def run(
    *,
    dataset=None,
    output_dir=None,
    threshold=None,
    crop_padding=None,
    image_size=None,
    detector_model_path=None,
    rematerialize=False,
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
    )
