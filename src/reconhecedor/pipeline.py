"""Orquestracao generica do fluxo da Spec 001 sobre as camadas da ADR 001.

Esta pasta contem apenas a **implementacao base** das camadas e a
orquestracao delas. A definicao de experimentos (configuracao, instanciação
das camadas e resultados) fica em ``src/experiments/<nome>/`` (ADR 002).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataloader.base import Dataloader
from .evaluator.evaluator import Evaluator
from .evaluator.metrics import Metrics
from .face_detector.base import FaceDetector
from .face_recognition.recognizer import FaceRecognition
from .model.base import Model
from .preprocessor.base import PreProcessor
from .types import PredictionRecord, Sample


@dataclass
class ExperimentResult:
    """Resultado de uma execucao de pipeline/experimento."""

    metrics: Metrics
    run_dir: Path
    n_gallery: int
    n_probe: int
    discarded_gallery: int
    discarded_probe: int


def _face_embedding(
    sample: Sample,
    *,
    detector: FaceDetector,
    preprocessor: PreProcessor,
    model: Model,
    padding: float,
) -> np.ndarray | None:
    """Spec 001/002: detecta a face, recorta, pre-processa e gera o embedding.

    Retorna ``None`` quando nao ha face (RF-02: imagem descartada).
    """
    detection = detector.detect(sample.image)
    if detection is None:
        return None
    face = detector.crop(sample.image, detection, padding=padding)
    if face is None or face.size == 0:
        return None
    return model.embed(preprocessor.process(face))


def run_experiment(
    *,
    name: str,
    gallery_loader: Dataloader,
    probe_loader: Dataloader,
    detector: FaceDetector,
    preprocessor: PreProcessor,
    model: Model,
    recognition: FaceRecognition | None = None,
    evaluator: Evaluator | None = None,
    dataset: str | Path = "",
    output_dir: str | Path,
    threshold: float,
    crop_padding: float,
) -> ExperimentResult:
    """Executa o pipeline completo (Spec 001) e persiste as metricas.

    As camadas chegam por injecao de dependencia: trocar a Model (ou qualquer
    outra camada) nao exige alteracoes neste pipeline (Spec 003 RNF-01).
    A configuracao do experimento (modelos, dataset, parametros) vem de
    ``src/experiments/<nome>/config.json`` (ADR 002); aqui apenas orquestramos
    a execucao e gravamos as metricas em ``output_dir`` (Spec 004 RNF-01).

    ``output_dir``, ``threshold`` e ``crop_padding`` sao **obrigatorios**: a
    base nao assume valores calibrados de experimento (ADR 002 — cada
    experimento passa os seus, tipicamente lidos do proprio ``config.json``;
    o resultado deve cair dentro de ``src/experiments/<nome>/results/``).
    """
    recognition = recognition if recognition is not None else FaceRecognition(threshold=threshold)
    evaluator = evaluator if evaluator is not None else Evaluator()

    # 1) Galeria: cadastra as pessoas do dominio (RF-03).
    gallery_samples = gallery_loader.load()
    discarded_gallery: list[Path] = []
    enrolled_ids: set[str] = set()
    for sample in gallery_samples:
        embedding = _face_embedding(
            sample, detector=detector, preprocessor=preprocessor, model=model, padding=crop_padding
        )
        if embedding is None:
            discarded_gallery.append(sample.path)
            continue
        recognition.enroll(sample.label, embedding)
        enrolled_ids.add(sample.label)

    # 2) Probe: reconhece ou classifica como desconhecido (RF-03/RF-04).
    probe_samples = probe_loader.load()
    discarded_probe: list[Path] = []
    records: list[PredictionRecord] = []
    for sample in probe_samples:
        embedding = _face_embedding(
            sample, detector=detector, preprocessor=preprocessor, model=model, padding=crop_padding
        )
        if embedding is None:
            discarded_probe.append(sample.path)  # RF-02
            continue
        result = recognition.identify(embedding)
        records.append(
            PredictionRecord(
                path=sample.path,
                true_label=sample.label,
                predicted_label=result.predicted_label,
                similarity=result.similarity,
            )
        )

    # 3) Avaliacao (Spec 004) + persistencia em arquivo (RNF-01).
    metrics = evaluator.evaluate(records, enrolled_ids)
    config_info = {
        "dataset": str(dataset),
        "model": model.name,
        "face_detector": detector.name,
        "preprocessor": preprocessor.name,
        "recognition": recognition.name,
        "threshold": threshold,
        "crop_padding": crop_padding,
    }
    counts = {
        "gallery_samples": len(gallery_samples),
        "gallery_enrolled": len(enrolled_ids),
        "probe_samples": len(probe_samples),
        "probe_evaluated": len(records),
        "discarded_no_face": len(discarded_gallery) + len(discarded_probe),
        "discarded_gallery": len(discarded_gallery),
        "discarded_probe": len(discarded_probe),
        **{key: value for key, value in metrics.to_dict().items() if key in ("tp", "fp", "tn", "fn")},
    }
    run_dir = evaluator.save(
        metrics,
        name=name,
        output_dir=output_dir,
        config=config_info,
        counts=counts,
    )

    for path in discarded_gallery + discarded_probe:
        print(f"[pipeline] descartada (sem face): {path}")

    return ExperimentResult(
        metrics=metrics,
        run_dir=run_dir,
        n_gallery=len(gallery_samples),
        n_probe=len(probe_samples),
        discarded_gallery=len(discarded_gallery),
        discarded_probe=len(discarded_probe),
    )
