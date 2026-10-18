"""Experimento baseline (ADR 002).

Instancia cada camada especificada na ADR 001 a partir de ``config.json``
desta subpasta, compondo uma instancia completa do sistema:

| Camada          | Implementação usada neste experimento                    |
|-----------------|----------------------------------------------------------|
| Dataloader      | ``FolderDataloader`` (``<dataset>/gallery`` e ``probe``)  |
| FaceDetector    | ``YuNetFaceDetector`` (``face_detection_model: yunet``)   |
| PreProcessor    | ``StandardPreProcessor`` (``image_size`` dos parâmetros)  |
| Model           | ``BaselineHogModel`` (``hog_model.py``; ``face_recognition_model``) |
| FaceRecognition  | ``FaceRecognition`` (cosseno; ``threshold`` dos parâmetros)|
| Evaluator        | ``Evaluator`` (métricas em ``results/``, RNF-01)           |

Tudo (inclusive a preparação do dataset em ``demo.py`` e o modelo HOG em
``hog_model.py``) vive nesta subpasta dedicada: nenhuma implementação de
experimento está em ``src/reconhecedor/`` (ADR 002, última característica).

Configuração replicável: ``config.json``. Resultados verificáveis:
``results/`` — ambos nesta subpasta (ADR 002, características 3 e 4).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

#: ``src/experiments/baseline/`` -> ``src/`` (permite importar sem instalar).
EXPERIMENT_DIR = Path(__file__).resolve().parent
_SRC_DIR = EXPERIMENT_DIR.parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from experiments.config import ExperimentConfig, load_config

from reconhecedor.dataloader import FolderDataloader
from reconhecedor.evaluator import Evaluator
from reconhecedor.face_detector import YuNetFaceDetector
from reconhecedor.face_recognition import FaceRecognition
from reconhecedor.pipeline import ExperimentResult, run_experiment
from reconhecedor.preprocessor import StandardPreProcessor

from .hog_model import BaselineHogModel

#: Implementações disponíveis para a camada FaceDetector (nome -> classe).
FACE_DETECTOR_MODELS = {"yunet": YuNetFaceDetector}
#: Implementações disponíveis para a camada Model (nome -> classe).
FACE_RECOGNITION_MODELS = {"baseline-hog": BaselineHogModel}

CONFIG_PATH = EXPERIMENT_DIR / "config.json"


def _required_param(config: ExperimentConfig, key: str) -> Any:
    """Le um parametro obrigatorio do ``config.json`` (ADR 002, característica 3).

    A base do sistema nao guarda valores calibrados do experimento (limiar,
    margem de recorte, diretorio de saida): eles precisam estar declarados
    aqui, na configuracao replicavel do experimento.
    """
    value = config.parameters.get(key)
    if value is None:
        raise ValueError(
            f"config.json do experimento '{config.name}' sem o parametro obrigatorio "
            f"'{key}' (ADR 002: configuracao replicavel do experimento)"
        )
    return value


def load() -> ExperimentConfig:
    """Carrega a configuração replicável deste experimento (ADR 002)."""
    return load_config(CONFIG_PATH)


def build_layers(
    config: ExperimentConfig,
    *,
    detector_model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Instancia as seis camadas da ADR 001 conforme a configuração.

    A troca de modelo (Spec 003 RNF-01) ocorre alterando o ``config.json``:
    nenhuma camada base é modificada.
    """
    if config.face_detection_model not in FACE_DETECTOR_MODELS:
        options = ", ".join(sorted(FACE_DETECTOR_MODELS))
        raise ValueError(
            f"detector '{config.face_detection_model}' desconhecido (disponiveis: {options})"
        )
    if config.face_recognition_model not in FACE_RECOGNITION_MODELS:
        options = ", ".join(sorted(FACE_RECOGNITION_MODELS))
        raise ValueError(
            f"model '{config.face_recognition_model}' desconhecida (disponiveis: {options})"
        )

    params = config.parameters
    image_size = int(params.get("image_size", 128))
    detector_kwargs: dict[str, Any] = {}
    path = detector_model_path if detector_model_path is not None else params.get("detector_model_path")
    if path:
        detector_kwargs["model_path"] = path

    return {
        "gallery_loader": FolderDataloader(config.gallery_root),
        "probe_loader": FolderDataloader(config.probe_root),
        "detector": FACE_DETECTOR_MODELS[config.face_detection_model](**detector_kwargs),
        "preprocessor": StandardPreProcessor(size=(image_size, image_size)),
        "model": FACE_RECOGNITION_MODELS[config.face_recognition_model](),
        "recognition": FaceRecognition(threshold=float(_required_param(config, "threshold"))),
        "evaluator": Evaluator(),
    }


def run(
    *,
    dataset: str | Path | None = None,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    crop_padding: float | None = None,
    image_size: int | None = None,
    detector_model_path: str | Path | None = None,
) -> ExperimentResult:
    """Executa o experimento ponta a ponta (Spec 001) e grava as métricas.

    Os argumentos são sobrescritas opcionais sobre o ``config.json``;
    ``threshold`` e ``crop_padding`` são obrigatórios na configuração (a base
    não guarda valores calibrados deste experimento — ADR 002). Por padrão os
    resultados vão para ``results/`` desta subpasta (ADR 002, característica 4;
    Spec 004 RNF-01).
    """
    config = load()
    if dataset is not None:
        config.dataset = Path(dataset)
    overrides: dict[str, Any] = {
        "threshold": threshold,
        "crop_padding": crop_padding,
        "image_size": image_size,
    }
    config.parameters.update({key: value for key, value in overrides.items() if value is not None})

    kwargs = build_layers(config, detector_model_path=detector_model_path)
    # Resultados dentro da subpasta do experimento (ADR 002, característica 4).
    results_dir = output_dir if output_dir is not None else config.results_dir
    return run_experiment(
        name=config.name,
        dataset=config.dataset,
        output_dir=results_dir,
        threshold=float(_required_param(config, "threshold")),
        crop_padding=float(_required_param(config, "crop_padding")),
        **kwargs,
    )


__all__ = ["CONFIG_PATH", "build_layers", "load", "run"]
