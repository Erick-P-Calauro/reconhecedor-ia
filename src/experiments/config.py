"""Configuração de experimento (ADR 002).

Definição de experimento: uma configuração de uma instância do sistema que
contém **modelo de detecção de face**, **modelo de reconhecimento de face**,
**dataset utilizado** e **parâmetros de configuração** (ADR 002). Cada
experimento guarda seu ``config.json`` em sua subpasta dedicada, para que sua
utilização seja replicada por qualquer usuário (ADR 002, característica 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Chaves obrigatorias de um ``config.json`` de experimento.
REQUIRED_KEYS = (
    "name",
    "dataset",
    "face_detection_model",
    "face_recognition_model",
    "parameters",
)


@dataclass
class ExperimentConfig:
    """Configuracao replicavel de um experimento (ADR 002)."""

    name: str
    dataset: str | Path
    face_detection_model: str
    face_recognition_model: str
    parameters: dict[str, Any] = field(default_factory=dict)
    #: parametros de dataset/split (Spec 005): tamanho da amostra, classes, etc.
    sample_size: int = 25
    n_classes: int = 15
    min_photos_per_class: int = 4
    validation_ratio: float = 0.5
    test_ratio: float = 0.5
    #: se True, resultados vao para ``<dir>/private/results`` (dataset privado, ADR 004).
    private_dataset: bool = False
    #: subpasta dedicada do experimento (``src/experiments/<nome>/``).
    directory: Path = Path(".")

    @property
    def gallery_root(self) -> Path:
        """Galeria do dataset (Spec 003: pessoas cadastradas)."""
        return Path(self.dataset) / "gallery"

    @property
    def probe_root(self) -> Path:
        """Sondas do dataset (Spec 003: fora da galeria = desconhecido)."""
        return Path(self.dataset) / "probe"

    @property
    def results_dir(self) -> Path:
        """Resultados dentro da subpasta do experimento (ADR 002, característica 4).

        Datasets privados (ADR 004) gravam em ``private/results`` (gitignored).
        """
        if self.private_dataset:
            return self.directory / "private" / "results"
        return self.directory / "results"


def load_config(path: str | Path) -> ExperimentConfig:
    """Carrega ``config.json`` de um experimento e valida suas chaves."""
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"configuracao de experimento nao encontrada: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"configuracao invalida em {path}: {exc}") from exc

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(f"configuracao {path} sem as chaves obrigatorias: {', '.join(missing)}")

    parameters = raw["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError(f"'parameters' deve ser um objeto em {path}")

    dataset_parameters = raw.get("dataset_parameters", {})
    if dataset_parameters is None:
        dataset_parameters = {}
    if not isinstance(dataset_parameters, dict):
        raise ValueError(f"'dataset_parameters' deve ser um objeto em {path}")

    sample_size = dataset_parameters.get("sample_size", 25)
    n_classes = dataset_parameters.get("n_classes", 15)
    min_photos_per_class = dataset_parameters.get("min_photos_per_class", 4)
    validation_ratio = dataset_parameters.get("validation_ratio", 0.5)
    if "test_ratio" in dataset_parameters and dataset_parameters.get("test_ratio") is not None:
        test_ratio = dataset_parameters.get("test_ratio")
    else:
        test_ratio = 1 - validation_ratio

    try:
        sample_size = int(sample_size)
        n_classes = int(n_classes)
        min_photos_per_class = int(min_photos_per_class)
        validation_ratio = float(validation_ratio)
        test_ratio = float(test_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'dataset_parameters' invalido em {path}: {exc}") from exc

    if sample_size <= 0:
        raise ValueError(f"'sample_size' deve ser > 0 em {path}")
    if n_classes <= 0:
        raise ValueError(f"'n_classes' deve ser > 0 em {path}")
    if min_photos_per_class < 1:
        raise ValueError(f"'min_photos_per_class' deve ser >= 1 em {path}")
    if not (0.0 <= validation_ratio <= 1.0):
        raise ValueError(f"'validation_ratio' deve estar em [0, 1] em {path}")
    if not (0.0 <= test_ratio <= 1.0):
        raise ValueError(f"'test_ratio' deve estar em [0, 1] em {path}")
    if abs(validation_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"'validation_ratio' + 'test_ratio' deve ser ~ 1.0 em {path}")
    if validation_ratio <= 0 or test_ratio <= 0:
        raise ValueError(f"'validation_ratio' e 'test_ratio' devem ser > 0 em {path}")

    return ExperimentConfig(
        name=str(raw["name"]),
        dataset=raw["dataset"],
        face_detection_model=str(raw["face_detection_model"]),
        face_recognition_model=str(raw["face_recognition_model"]),
        parameters=dict(parameters),
        sample_size=sample_size,
        n_classes=n_classes,
        min_photos_per_class=min_photos_per_class,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        private_dataset=bool(raw.get("private_dataset", False)),
        directory=path.parent,
    )


__all__ = ["ExperimentConfig", "load_config", "REQUIRED_KEYS"]
