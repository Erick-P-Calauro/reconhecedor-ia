"""Testes das metricas do experimento (Spec 004 RF-02..RF-06 e RNF-01)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reconhecedor.evaluator import Evaluator, Metrics, compute_metrics
from reconhecedor.types import PredictionRecord


def _record(true: str, predicted: str | None, similarity: float = 0.9) -> PredictionRecord:
    return PredictionRecord(
        path=Path(f"{true}.png"),
        true_label=true,
        predicted_label=predicted,
        similarity=similarity,
    )


def _sample_records() -> list[PredictionRecord]:
    # Domínio: "alice". Enroladas: 2 corretas, 1 recusada (FN).
    # Desconhecidas ("bob"): 1 aceita (FP), 1 recusada (TN).
    return [
        _record("alice", "alice"),
        _record("alice", "alice"),
        _record("alice", None, 0.3),
        _record("bob", "alice", 0.7),
        _record("bob", None, 0.2),
    ]


def test_metrics_values():
    metrics = compute_metrics(_sample_records(), enrolled_labels={"alice"})
    assert isinstance(metrics, Metrics)
    assert (metrics.tp, metrics.fp, metrics.tn, metrics.fn) == (2, 1, 1, 1)
    assert metrics.total == 5
    assert metrics.accuracy == 0.6  # (TP + TN) / 5
    assert round(metrics.precision, 6) == round(2 / 3, 6)  # RF-03
    assert round(metrics.recall, 6) == round(2 / 3, 6)  # RF-04
    assert metrics.far == 0.5  # RF-05: 1 / (1 + 1)
    assert round(metrics.frr, 6) == round(1 / 3, 6)  # RF-06: 1 / (1 + 2)


def test_wrong_identity_counts_as_false_negative():
    records = [_record("alice", "bob")]
    metrics = compute_metrics(records, enrolled_labels={"alice", "bob"})
    assert metrics.tp == 0 and metrics.fn == 1
    assert metrics.recall == 0.0 and metrics.frr == 1.0


def test_empty_denominators_are_zero():
    # Sem desconhecidos: FAR = 0 (divisao por zero evitada).
    only_enrolled = [_record("alice", "alice")]
    metrics = compute_metrics(only_enrolled, enrolled_labels={"alice"})
    assert metrics.far == 0.0 and metrics.frr == 0.0
    assert metrics.accuracy == 1.0
    # Sem predicoes: tudo zerado.
    empty = compute_metrics([], enrolled_labels=set())
    assert empty.total == 0
    assert (empty.accuracy, empty.precision, empty.recall) == (0.0, 0.0, 0.0)
    assert (empty.far, empty.frr) == (0.0, 0.0)


def test_save_writes_json_and_history(tmp_path: Path):
    evaluator = Evaluator()
    metrics = evaluator.evaluate(_sample_records(), {"alice"})
    run_dir = evaluator.save(
        metrics,
        name="exp-a",
        output_dir=tmp_path,
        config={"dataset": "data/lfw", "model": "resnet50"},
        counts={"discarded_no_face": 3},
    )

    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert report["experiment"]["name"] == "exp-a"
    assert report["config"]["model"] == "resnet50"
    assert report["counts"]["discarded_no_face"] == 3
    for key in ("accuracy", "precision", "recall", "far", "frr"):
        assert key in report["metrics"]

    history = tmp_path / "metrics_history.csv"
    assert history.is_file()
    # Segunda execucao faz append (consulta posterior de varios experimentos).
    evaluator.save(metrics, name="exp-b", output_dir=tmp_path)
    with history.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["exp-a", "exp-b"]
    assert float(rows[0]["accuracy"]) == 0.6
