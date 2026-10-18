"""Experimento baseline: detector YuNet + Model HOG sobre o dataset demo.

Configuração em ``config.json``; wiring em ``experiment.py``; preparação do
dataset em ``demo.py`` (``build-demo``); modelo HOG em ``hog_model.py``;
resultados em ``results/`` (ADR 002).
"""

from .experiment import CONFIG_PATH, build_layers, load, run

__all__ = ["CONFIG_PATH", "build_layers", "load", "run"]
