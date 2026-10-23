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
    validation_metrics: Metrics | None = None
    n_validation: int = 0
    n_test: int = 0
    discarded_validation: int = 0
    discarded_test: int = 0


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
def _process_loader(loader, *, detector, preprocessor, model, recognition, crop_padding):
    """Processa um loader em (samples, records, discarded)."""
    if loader is None:
        return [], [], []
    samples = loader.load()
    discarded: list[Path] = []
    records: list[PredictionRecord] = []
    for sample in samples:
        embedding = _face_embedding(
            sample, detector=detector, preprocessor=preprocessor, model=model, padding=crop_padding
        )
        if embedding is None:
            discarded.append(sample.path)
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
    return samples, records, discarded


def run_experiment(
    *,
    name: str,
    gallery_loader: Dataloader,
    probe_loader: Dataloader | None = None,
    validation_loader: Dataloader | None = None,
    test_loader: Dataloader | None = None,
    detector: FaceDetector,
    preprocessor: PreProcessor,
    model: Model,
    recognition: FaceRecognition | None = None,
    evaluator: Evaluator | None = None,
    dataset: str | Path = "",
    output_dir: str | Path,
    threshold: float,
    crop_padding: float,
    sample_size=None,
    n_classes=None,
    min_photos_per_class=None,
    validation_ratio=None,
    test_ratio=None,
) -> ExperimentResult:
    
    recognition = recognition if recognition is not None else FaceRecognition(threshold=threshold)
    evaluator = evaluator if evaluator is not None else Evaluator()

    if test_loader is None:
        test_loader = probe_loader
    has_validation = validation_loader is not None

    # 1) Galeria: cadastra as pessoas do dominio.
    print("[pipeline] carregando galeria")
    gallery_samples = gallery_loader.load()
    print(f"[pipeline] galeria: {len(gallery_samples)} imagens")
    discarded_gallery: list[Path] = []
    enrolled_ids: set[str] = set()
    print("[pipeline] detectando faces na galeria")
    for sample in gallery_samples:
        embedding = _face_embedding(
            sample, detector=detector, preprocessor=preprocessor, model=model, padding=crop_padding
        )
        if embedding is None:
            discarded_gallery.append(sample.path)
            continue
        recognition.enroll(sample.label, embedding)
        enrolled_ids.add(sample.label)
    print(f"[pipeline] galeria concluida: {len(enrolled_ids)} identidades, {len(discarded_gallery)} descartadas")

    # 2) Validacao (opcional).
    validation_samples: list = []
    validation_records: list[PredictionRecord] = []
    discarded_validation: list[Path] = []
    validation_metrics: Metrics | None = None
    if has_validation:
        print("[pipeline] carregando validacao")
        validation_samples, validation_records, discarded_validation = _process_loader(
            validation_loader,
            detector=detector,
            preprocessor=preprocessor,
            model=model,
            recognition=recognition,
            crop_padding=crop_padding,
        )
        print(f"[pipeline] validacao: {len(validation_samples)} imagens")
        print(f"[pipeline] validacao concluida: {len(validation_records)} avaliadas, {len(discarded_validation)} descartadas")
        print("[pipeline] avaliando validacao")
        validation_metrics = evaluator.evaluate(validation_records, enrolled_ids)

    # 3) Teste.
    print("[pipeline] carregando teste")
    test_samples, test_records, discarded_test = _process_loader(
        test_loader,
        detector=detector,
        preprocessor=preprocessor,
        model=model,
        recognition=recognition,
        crop_padding=crop_padding,
    )
    print(f"[pipeline] teste: {len(test_samples)} imagens")
    print(f"[pipeline] teste concluido: {len(test_records)} avaliadas, {len(discarded_test)} descartadas")

    # 4) Avaliacao (Spec 004) + persistencia em arquivo.
    print("[pipeline] avaliando e persistindo")
    metrics = evaluator.evaluate(test_records, enrolled_ids)
    config_info = {
        "dataset": str(dataset),
        "model": model.name,
        "face_detector": detector.name,
        "preprocessor": preprocessor.name,
        "recognition": recognition.name,
        "threshold": threshold,
        "crop_padding": crop_padding,
        "sample_size": sample_size,
        "n_classes": n_classes,
        "min_photos_per_class": min_photos_per_class,
        "validation_ratio": validation_ratio,
        "test_ratio": test_ratio,
        "dataset_parameters": {
            "sample_size": sample_size,
            "n_classes": n_classes,
            "min_photos_per_class": min_photos_per_class,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
        },
    }
    n_validation = len(validation_samples)
    n_test = len(test_samples)
    # n_probe = soma validacao+teste (ou probe, quando nao houver split)
    n_probe = n_validation + n_test

    discarded_probe_compat = len(discarded_test)
    counts = {
        "gallery_samples": len(gallery_samples),
        "gallery_enrolled": len(enrolled_ids),
        "probe_samples": n_probe,
        "validation_samples": n_validation,
        "validation_evaluated": len(validation_records),
        "test_samples": n_test,
        "probe_evaluated": len(test_records),
        "discarded_no_face": len(discarded_gallery) + len(discarded_test) + len(discarded_validation),
        "discarded_gallery": len(discarded_gallery),
        "discarded_probe": discarded_probe_compat,
        "discarded_validation": len(discarded_validation),
        "discarded_test": len(discarded_test),
        **{key: value for key, value in metrics.to_dict().items() if key in ("tp", "fp", "tn", "fn")},
    }
    validation_counts = {
        "validation_samples": n_validation,
        "validation_evaluated": len(validation_records),
        "discarded_validation": len(discarded_validation),
    }
    run_dir = evaluator.save(
        metrics,
        name=name,
        output_dir=output_dir,
        config=config_info,
        counts=counts,
        validation_metrics=validation_metrics,
        validation_counts=validation_counts if has_validation else None,
    )

    all_discarded = discarded_gallery + discarded_validation + discarded_test
    for path in all_discarded:
        print(f"[pipeline] descartada (sem face): {path}")

    return ExperimentResult(
        metrics=metrics,
        run_dir=run_dir,
        n_gallery=len(gallery_samples),
        n_probe=n_probe,
        discarded_gallery=len(discarded_gallery),
        discarded_probe=discarded_probe_compat,
        validation_metrics=validation_metrics,
        n_validation=n_validation,
        n_test=n_test,
        discarded_validation=len(discarded_validation),
        discarded_test=len(discarded_test),
    )
