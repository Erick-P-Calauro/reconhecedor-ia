"""Teste de integracao: pipeline ponta a ponta (Spec 001 + ADR 001).

Requer o dataset demo; se ausente, o teste e pulado com instrucao de como
gera-lo (``python main.py build-demo``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reconhecedor.dataloader import FolderDataloader
from reconhecedor.evaluator import Evaluator
from reconhecedor.face_detector import YuNetFaceDetector
from reconhecedor.face_recognition import FaceRecognition
from reconhecedor.model import BaselineHogModel
from reconhecedor.pipeline import ExperimentConfig, run_experiment
from reconhecedor.preprocessor import StandardPreProcessor

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "demo"

pytestmark = pytest.mark.skipif(
    not (DEMO / "gallery").is_dir() or not (DEMO / "probe").is_dir(),
    reason="dataset demo ausente: rode 'python main.py build-demo'",
)


@pytest.fixture(scope="module")
def experiment_result(tmp_path_factory: pytest.TempPathFactory):
    output_dir = tmp_path_factory.mktemp("results")
    config = ExperimentConfig(
        name="teste-integracao",
        dataset_root=DEMO,
        output_dir=output_dir,
        threshold=0.8,
    )
    result = run_experiment(
        config,
        gallery_loader=FolderDataloader(config.gallery_root),
        probe_loader=FolderDataloader(config.probe_root),
        detector=YuNetFaceDetector(),
        preprocessor=StandardPreProcessor(size=(128, 128)),
        model=BaselineHogModel(),
        recognition=FaceRecognition(threshold=config.threshold),
        evaluator=Evaluator(),
    )
    return config, result


def test_pipeline_discards_images_without_face(experiment_result):
    """Spec 002 RF-02: imagens sem face nao entram na avaliacao."""
    _, result = experiment_result
    assert result.n_probe == 6  # 3 astronaut + 1 lena + 2 sem face
    assert result.discarded_probe == 2  # coffee.png e chelsea.png
    assert result.metrics.total == 4  # apenas imagens com face avaliadas


def test_pipeline_recognizes_enrolled_and_rejects_unknown(experiment_result):
    """Spec 003 RF-03/RF-04 + Spec 004: metricas coerentes."""
    _, result = experiment_result
    metrics = result.metrics
    assert metrics.total > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.accuracy >= 0.6  # genuine aceitas + imposta recusada
    # A desconhecida (lena) deve ser rejeitada (RF-04).
    assert metrics.fp == 0, "pessoa desconhecida foi aceita"


def test_pipeline_persists_metrics_file(experiment_result):
    """Spec 004 RNF-01: metricas mantidas em arquivo para consulta posterior."""
    config, result = experiment_result
    report_path = result.run_dir / "metrics.json"
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["config"]["dataset"] == str(config.dataset_root)
    assert report["config"]["model"] == "baseline-hog"
    assert report["counts"]["discarded_no_face"] == 2
    for key in ("accuracy", "precision", "recall", "far", "frr"):
        assert key in report["metrics"]
    assert (Path(config.output_dir) / "metrics_history.csv").is_file()
