# Orquestração genérica do fluxo de reconhecimento facial e das camadas base do sistema
# A definição dos experimentos fica em src/experiments

from __future__ import annotations
import numpy as np
from .dataloader.base import Dataloader
from .evaluator.evaluator import Evaluator
from .evaluator.metrics import Metrics
from .face_detector.base import FaceDetector
from .face_recognition.recognizer import FaceRecognition
from .model.base import Model
from .preprocessor.base import PreProcessor
from .types import PredictionRecord, Sample
from dataclasses import dataclass
from pathlib import Path

@dataclass
class ExperimentResult:
    metrics: Metrics
    run_dir: Path
    n_gallery: int
    n_probe: int
    discarded_gallery: int
    discarded_probe: int


# Detecta a face, recorta, pré-processa e gera embedding
# FaceDetector, PreProcessor e Model devem ser substituídos em cada experimento
def _face_embedding(
    sample: Sample,
    *,
    detector: FaceDetector,
    preprocessor: PreProcessor,
    model: Model,
    padding: float,
) -> np.ndarray | None:
    detection = detector.detect(sample.image)
    if detection is None:
        return None
    face = detector.crop(sample.image, detection, padding=padding)
    if face is None or face.size == 0:
        return None
    return model.embed(preprocessor.process(face))

# Executa o pipeline completo com camadas recebidas por injeção de dependências
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
    
    recognition = recognition if recognition is not None else FaceRecognition(threshold=threshold)
    evaluator = evaluator if evaluator is not None else Evaluator()

    # 1) Galeria: cadastra as pessoas do dominio (RF-03).
    print("[pipeline] carregando galeria")
    gallery_samples = gallery_loader.load()
    print(f"[pipeline] galeria: {len(gallery_samples)} imagens")
    discarded_gallery: list[Path] = []
    enrolled_ids: set[str] = set()
    print("[pipeline] detectando faces na galeria")
    for index, sample in enumerate(gallery_samples, 1):
        embedding = _face_embedding(
            sample, detector=detector, preprocessor=preprocessor, model=model, padding=crop_padding
        )
        if embedding is None:
            discarded_gallery.append(sample.path)
            continue
        recognition.enroll(sample.label, embedding)
        enrolled_ids.add(sample.label)
    print(f"[pipeline] galeria concluida: {len(enrolled_ids)} identidades, {len(discarded_gallery)} descartadas")

    # 2) Probe: reconhece ou classifica como desconhecido (RF-03/RF-04).
    print("[pipeline] carregando probe")
    probe_samples = probe_loader.load()
    print(f"[pipeline] probe: {len(probe_samples)} imagens")
    discarded_probe: list[Path] = []
    records: list[PredictionRecord] = []
    print("[pipeline] detectando/identificando probe")
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
    print(f"[pipeline] probe concluido: {len(records)} avaliadas, {len(discarded_probe)} descartadas")

    # 3) Avaliacao (Spec 004) + persistencia em arquivo (RNF-01).
    print("[pipeline] avaliando e persistindo")
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
