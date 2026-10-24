"""Persistencia das metricas por experimento (Spec 004)."""

from __future__ import annotations
import csv
import json
from datetime import datetime
from pathlib import Path
from ..types import PredictionRecord
from .metrics import Metrics, compute_metrics

HISTORY_FILENAME = "metrics_history.csv"
HISTORY_FIELDS = [
    "timestamp",
    "experiment",
    "dataset",
    "model",
    "accuracy",
    "precision",
    "recall",
    "far",
    "frr",
    "tp",
    "fp",
    "tn",
    "fn",
    "discarded_no_face",
]


class Evaluator:
    """Camada responsavel por gerar metricas e avaliar um experimento."""

    def __init__(self) -> None:
        print("[Evaluator] inicializando")
        print("[Evaluator] inicializado")

    def evaluate(
        self,
        records: list[PredictionRecord],
        enrolled_labels: list[str] | set[str],
    ) -> Metrics:
        """Gera as metricas de um experimento."""
        print("[Evaluator] avaliando")
        metrics = compute_metrics(records, enrolled_labels)
        print("[Evaluator] avaliacao concluida")
        return metrics

    def save(
        self,
        metrics: Metrics,
        *,
        name: str,
        output_dir: str | Path,
        config: dict | None = None,
        counts: dict | None = None,
        validation_metrics: Metrics | None = None,
        validation_counts: dict | None = None,
        threshold_used=None,
        optimization_info=None,
    ) -> Path:
        """Grava as metricas em arquivo para consulta posterior."""
        print("[Evaluator] salvando resultados")
        output_dir = Path(output_dir)
        timestamp = datetime.now()
        run_dir = output_dir / name / timestamp.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "experiment": {"name": name, "timestamp": timestamp.isoformat(timespec="seconds")},
            "config": config or {},
            "counts": counts or {},
            "metrics": metrics.to_dict(),
        }
        if validation_metrics is not None:
            payload["validation_metrics"] = validation_metrics.to_dict()
        if validation_counts is not None:
            payload["validation_counts"] = validation_counts
        if threshold_used is not None:
            payload["threshold_used"] = threshold_used
        if optimization_info is not None:
            payload["optimization_info"] = optimization_info
        (run_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._append_history(output_dir / HISTORY_FILENAME, payload)
        print(f"[Evaluator] resultados salvos em {run_dir}")
        return run_dir

    @staticmethod
    def _append_history(history_path: Path, payload: dict) -> None:
        counts = payload.get("counts", {})
        config = payload.get("config", {})
        row = {
            "timestamp": payload["experiment"]["timestamp"],
            "experiment": payload["experiment"]["name"],
            "dataset": config.get("dataset", ""),
            "model": config.get("model", ""),
            **{key: payload["metrics"][key] for key in ("accuracy", "precision", "recall", "far", "frr", "tp", "fp", "tn", "fn")},
            "discarded_no_face": counts.get("discarded_no_face", 0),
        }
        history_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not history_path.is_file()
        with history_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
