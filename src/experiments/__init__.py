"""Experimentos do sistema (ADR 002).

Cada experimento vive em uma subpasta dedicada dentro de ``src/experiments/``,
contendo:

- ``config.json``    — configuracao replicavel (modelos, dataset, parametros);
- ``experiment.py``  — instanciação de cada camada da ADR 001 (wiring);
- ``results/``       — resultados verificaveis da replicacao (Spec 004).

Nenhuma implementacao de experimento fica em ``src/reconhecedor/`` (pasta base
das camadas da ADR 001).
"""

from __future__ import annotations

from pathlib import Path

#: ``src/experiments/`` — raiz de todos os experimentos.
EXPERIMENTS_ROOT = Path(__file__).resolve().parent


def experiment_dir(name: str) -> Path:
    """Subpasta dedicada do experimento ``name``."""
    return EXPERIMENTS_ROOT / name


def discover_experiments() -> list[str]:
    """Lista os experimentos disponiveis (ADR 002, caracteristica 2).

    Um diretorio qualifica como experimento quando possui ``config.json``
    (configuracao) e ``experiment.py`` (instanciação das camadas).
    """
    names: list[str] = []
    for entry in sorted(EXPERIMENTS_ROOT.iterdir()):
        if not entry.is_dir() or entry.name.startswith(("_", ".")):
            continue
        if (entry / "config.json").is_file() and (entry / "experiment.py").is_file():
            names.append(entry.name)
    return names


__all__ = ["EXPERIMENTS_ROOT", "discover_experiments", "experiment_dir"]
