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

    return ExperimentConfig(
        name=str(raw["name"]),
        dataset=raw["dataset"],
        face_detection_model=str(raw["face_detection_model"]),
        face_recognition_model=str(raw["face_recognition_model"]),
        parameters=dict(parameters),
        private_dataset=bool(raw.get("private_dataset", False)),
        directory=path.parent,
    )


__all__ = ["ExperimentConfig", "load_config", "REQUIRED_KEYS"]
