"""Otimizacao de threshold por F1 (Spec 005).

Busca o threshold de similaridade que maximiza o F1 sobre um conjunto de
validacao, usando ``best_label`` + ``similarity`` de cada ``PredictionRecord``.
"""

from __future__ import annotations

from typing import Any


class ThresholdOptimizationResult(float):
    """Resultado da otimizacao; comporta-se como o threshold (float)."""

    def __new__(
        cls,
        threshold: float,
        *,
        f1: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        tp: int = 0,
        fp: int = 0,
        tn: int = 0,
        fn: int = 0,
    ):
        obj = super().__new__(cls, float(threshold))
        obj._f1 = float(f1)
        obj._precision = float(precision)
        obj._recall = float(recall)
        obj._tp = int(tp)
        obj._fp = int(fp)
        obj._tn = int(tn)
        obj._fn = int(fn)
        obj._data = {
            "threshold": float(threshold),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        }
        return obj

    @property
    def threshold(self) -> float:
        return float(self)

    @property
    def f1(self) -> float:
        return self._f1

    @property
    def precision(self) -> float:
        return self._precision

    @property
    def recall(self) -> float:
        return self._recall

    @property
    def tp(self) -> int:
        return self._tp

    @property
    def fp(self) -> int:
        return self._fp

    @property
    def tn(self) -> int:
        return self._tn

    @property
    def fn(self) -> int:
        return self._fn

    @property
    def info(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self),
            "f1": self._f1,
            "precision": self._precision,
            "recall": self._recall,
            "tp": self._tp,
            "fp": self._fp,
            "tn": self._tn,
            "fn": self._fn,
        }

    def __getitem__(self, key):
        data = self.to_dict()
        if isinstance(key, int):
            values = (float(self), data)
            return values[key]
        return data[key]

    def get(self, key, default=None):
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def values(self):
        return self.to_dict().values()

    def items(self):
        return self.to_dict().items()

    def __iter__(self):
        yield float(self)
        yield self.to_dict()


class ThresholdOptimizer:
    """Busca o threshold que maximiza o F1 em um conjunto de validacao."""

    def __init__(self) -> None:
        print("[ThresholdOptimizer] inicializando")
        print("[ThresholdOptimizer] inicializado")

    def optimize(self, records, enrolled_labels=None, **kwargs):
        if enrolled_labels is None:
            enrolled_labels = kwargs.get("enrolled") or kwargs.get("enrolled_ids")
        enrolled = set(enrolled_labels) if enrolled_labels is not None else set()

        if not records or not enrolled:
            print("[ThresholdOptimizer] sem dados/enrolled para otimizar; threshold nao alterado")
            return None

        candidates = {0.0, 1.0}
        for record in records:
            candidates.add(float(getattr(record, "similarity", 0.0)))
        # midpoints para cobrir mudancas de decisao entre scores
        sorted_scores = sorted(candidates)
        for left, right in zip(sorted_scores, sorted_scores[1:]):
            candidates.add((left + right) / 2.0)

        best_threshold = 0.0
        best_f1 = -1.0
        best_stats = (0, 0, 0, 0, 0.0, 0.0)
        for threshold in sorted(candidates):
            tp = fp = tn = fn = 0
            for record in records:
                true_label = getattr(record, "true_label", None)
                best_label = getattr(record, "best_label", None)
                similarity = float(getattr(record, "similarity", 0.0))
                is_enrolled = true_label in enrolled if enrolled else True
                accepted = similarity >= threshold
                if is_enrolled:
                    if accepted and best_label == true_label:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if accepted:
                        fp += 1
                    else:
                        tn += 1
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_stats = (tp, fp, tn, fn, precision, recall)

        tp, fp, tn, fn, precision, recall = best_stats
        return ThresholdOptimizationResult(
            best_threshold,
            f1=best_f1 if best_f1 >= 0 else 0.0,
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )

    def find_best_threshold(self, records, enrolled_labels=None, **kwargs):
        return self.optimize(records, enrolled_labels=enrolled_labels, **kwargs)

    def optimize_threshold(self, records, enrolled_labels=None, **kwargs):
        return self.optimize(records, enrolled_labels=enrolled_labels, **kwargs)


__all__ = ["ThresholdOptimizer", "ThresholdOptimizationResult"]