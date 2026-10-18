"""Metricas de avaliacao de experimentos (Spec 004 RF-02..RF-06).

Definicoes binarias "aceitar/recusar" sobre cada imagem de probe:

- ``TP`` (verdadeiro positivo): pessoa cadastrada, reconhecida corretamente.
- ``FN`` (falso negativo): pessoa cadastrada, recusada ou reconhecida com a
  identidade errada  -> FRR.
- ``FP`` (falso positivo): pessoa desconhecida aceita como alguem do dominio
  -> FAR.
- ``TN`` (verdadeiro negativo): pessoa desconhecida recusada.

- acurácia  = (TP + TN) / total          (RF-02)
- precisao  = TP / (TP + FP)             (RF-03)
- recall    = TP / (TP + FN)             (RF-04)
- FAR       = FP / (FP + TN)             (RF-05)
- FRR       = FN / (FN + TP)             (RF-06)

Divisao por zero (conjunto vazio) resulta em 0.0, com o total zerado na
contagem correspondente.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..types import PredictionRecord


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    far: float
    frr: float
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def to_dict(self) -> dict:
        data = asdict(self)
        data["total"] = self.total
        return data


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(
    records: list[PredictionRecord],
    enrolled_labels: list[str] | set[str],
) -> Metrics:
    """Calcula as metricas da Spec 004 a partir das predicoes do experimento.

    ``enrolled_labels`` define quem esta dentro do dominio (galeria): probes
    com identidade fora dessa lista sao desconhecidos (verdade negativa).
    """
    enrolled = set(enrolled_labels)
    tp = fp = tn = fn = 0

    for record in records:
        is_enrolled = record.true_label in enrolled
        accepted = record.predicted_label is not None
        if is_enrolled:
            if accepted and record.predicted_label == record.true_label:
                tp += 1
            else:
                fn += 1
        else:
            if accepted:
                fp += 1
            else:
                tn += 1

    total = tp + fp + tn + fn
    return Metrics(
        accuracy=round(_safe_divide(tp + tn, total), 6),
        precision=round(_safe_divide(tp, tp + fp), 6),
        recall=round(_safe_divide(tp, tp + fn), 6),
        far=round(_safe_divide(fp, fp + tn), 6),
        frr=round(_safe_divide(fn, fn + tp), 6),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
