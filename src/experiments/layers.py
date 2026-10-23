"""Wiring generico das seis camadas da ADR 001 para os experimentos do ADR 003.

Concentra a construcao das camadas a partir dos **nomes** declarados no
``config.json`` (``face_detection_model`` / ``face_recognition_model``), de modo
que trocar de modelo/detector seja apenas questao de editar a configuracao
(Spec 003, ADR 002).

Vive no pacote de experimentos (e nao na base ``src/reconhecedor/``) por ser
*wiring* de experimento; as classes concretas usadas sao genericas e ficam na
base (ADR 002). Os tres ``experiment.py`` do ADR 003 delegam para este modulo,
evitando triplicar o mesmo wiring.
"""

from __future__ import annotations

from pathlib import Path

from experiments.config import ExperimentConfig, load_config
from reconhecedor.dataloader import build_lfw_loaders, lfw_dataset
from reconhecedor.evaluator import Evaluator
from reconhecedor.face_detector import FaceNetDetector, YuNetFaceDetector
from reconhecedor.face_recognition import FaceRecognition
from reconhecedor.model import FaceNetModel, ResNet50Model
from reconhecedor.pipeline import ExperimentResult, run_experiment
from reconhecedor.preprocessor import TorchPreProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FACENET_MEAN = (0.5, 0.5, 0.5)
FACENET_STD = (0.5, 0.5, 0.5)


def build_detector(name: str, *, detector_model_path: str | Path | None = None):
    """Instancia o detector de face pelo nome declarado na configuracao."""
    if name == "yunet":
        if detector_model_path is not None:
            return YuNetFaceDetector(model_path=detector_model_path)
        return YuNetFaceDetector()
    if name == "facenet":
        return FaceNetDetector()
    raise ValueError(f"face_detection_model desconhecido: {name!r} (use 'yunet' ou 'facenet')")


def build_recognition(name: str, *, image_size: int):
    """Instancia (Model, PreProcessor) casados pelo nome do modelo de reconhecimento."""
    if name == "resnet50":
        model = ResNet50Model()
        preprocessor = TorchPreProcessor(
            size=(image_size, image_size), mean=IMAGENET_MEAN, std=IMAGENET_STD, name="imagenet"
        )
        return model, preprocessor
    if name == "facenet":
        model = FaceNetModel()
        preprocessor = TorchPreProcessor(
            size=(image_size, image_size), mean=FACENET_MEAN, std=FACENET_STD, name="facenet"
        )
        return model, preprocessor
    raise ValueError(f"face_recognition_model desconhecido: {name!r} (use 'resnet50' ou 'facenet')")


def build_layers(
    config: ExperimentConfig,
    *,
    detector_model_path: str | Path | None = None,
    rematerialize: bool = False,
) -> dict:
    """Monta o dicionario de camadas da ADR 001 exigido pelo pipeline."""
    params = config.parameters
    detector = build_detector(config.face_detection_model, detector_model_path=detector_model_path)
    model, preprocessor = build_recognition(
        config.face_recognition_model, image_size=int(params["image_size"])
    )
    gallery_loader, probe_loader = build_lfw_loaders(
        config.dataset,
        rematerialize=rematerialize,
    )
    return {
        "gallery_loader": gallery_loader,
        "probe_loader": probe_loader,
        "detector": detector,
        "preprocessor": preprocessor,
        "model": model,
        "recognition": FaceRecognition(float(params["threshold"])),
        "evaluator": Evaluator(),
    }


def run(
    config_path: str | Path,
    *,
    dataset: str | Path | None = None,
    output_dir: str | Path | None = None,
    threshold: float | None = None,
    crop_padding: float | None = None,
    image_size: int | None = None,
    detector_model_path: str | Path | None = None,
    rematerialize: bool = False,
) -> ExperimentResult:
    """Executa um experimento ponta a ponta a partir do seu ``config.json``.

    Os argumentos opcionais sobrescrevem a configuracao; os resultados sao
    gravados em ``config.results_dir`` (subpasta do experimento, ADR 002).
    """
    config = load_config(config_path)
    if dataset is not None:
        config.dataset = dataset
    if threshold is not None:
        config.parameters["threshold"] = threshold
    if crop_padding is not None:
        config.parameters["crop_padding"] = crop_padding
    if image_size is not None:
        config.parameters["image_size"] = image_size

    # O dataset LFW e materializado sob demanda.
    lfw_dataset.ensure_dataset(config.dataset, rematerialize=rematerialize)

    layers = build_layers(config, detector_model_path=detector_model_path, rematerialize=False)
    return run_experiment(
        name=config.name,
        dataset=str(config.dataset),
        output_dir=output_dir or config.results_dir,
        threshold=float(config.parameters["threshold"]),
        crop_padding=float(config.parameters["crop_padding"]),
        **layers,
    )
