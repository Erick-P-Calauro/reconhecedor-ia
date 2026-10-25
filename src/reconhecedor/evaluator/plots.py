"""Geracao de graficos de avaliacao (histograma, ROC, FAR/FRR x threshold)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore


def _to_arrays(records: Iterable[Any], enrolled: set[str]):
    sims = []
    labels = []  # 1 = genuino (cadastrado), 0 = impostor (desconhecido)
    enrolled_flags = []
    for r in records or []:
        sim = float(getattr(r, "similarity", 0.0))
        true_label = getattr(r, "true_label", None)
        is_enrolled = true_label in enrolled
        sims.append(sim)
        labels.append(1 if is_enrolled else 0)
        enrolled_flags.append(is_enrolled)
    return np.asarray(sims, dtype=float), np.asarray(labels, dtype=int), enrolled_flags


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_plots(
    records: Iterable[Any],
    enrolled_labels: Iterable[str] | None = None,
    output_dir: str | Path = ".",
    *,
    threshold_used: float | None = None,
    aliases: bool = True,
) -> list[Path]:
    """Gera e salva os graficos de avaliacao em ``output_dir``.

    Retorna a lista de arquivos criados.
    """
    if plt is None:
        return []

    enrolled = set(enrolled_labels or [])
    sims, labels, enrolled_flags = _to_arrays(records, enrolled)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # ---------- Histograma de similaridades ----------
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        gen = sims[labels == 1]
        imp = sims[labels == 0]
        bins = np.linspace(0.0, 1.0, 21)
        if len(gen):
            ax.hist(gen, bins=bins, alpha=0.6, label="Genuinos (cadastrados)", color="#1f77b4")
        if len(imp):
            ax.hist(imp, bins=bins, alpha=0.6, label="Impostores (desconhecidos)", color="#d62728")
        if threshold_used is not None:
            ax.axvline(float(threshold_used), color="black", linestyle="--", label=f"threshold={threshold_used:.3f}")
        ax.set_xlabel("Similaridade")
        ax.set_ylabel("Frequencia")
        ax.set_title("Histograma de similaridades")
        ax.legend()
        ax.grid(True, alpha=0.3)
        primary = output_dir / "histograma_similaridade.png"
        _save(fig, primary)
        created.append(primary)
        if aliases:
            for name in ("histogram.png", "histograma.png", "similarity_histogram.png", "similarities_histogram.png"):
                (output_dir / name).write_bytes(primary.read_bytes())
                created.append(output_dir / name)
    except Exception:
        pass

    # ---------- Curva ROC ----------
    try:
        from sklearn.metrics import roc_curve, auc

        if len(np.unique(labels)) >= 2:
            fpr, tpr, _ = roc_curve(labels, sims)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("FPR (FAR)")
            ax.set_ylabel("TPR (1-FRR)")
            ax.set_title("Curva ROC")
            ax.legend()
            ax.grid(True, alpha=0.3)
            primary = output_dir / "curva_roc.png"
            _save(fig, primary)
            created.append(primary)
            if aliases:
                for name in ("roc.png", "roc_curve.png", "roc_curve.png"):
                    (output_dir / name).write_bytes(primary.read_bytes())
                    created.append(output_dir / name)
    except Exception:
        pass

    # ---------- FAR/FRR em funcao do threshold ----------
    try:
        thresholds = np.linspace(0.0, 1.0, 101)
        fars = []
        frrs = []
        for t in thresholds:
            tp = fp = tn = fn = 0
            for r in records or []:
                sim = float(getattr(r, "similarity", 0.0))
                true_label = getattr(r, "true_label", None)
                best_label = getattr(r, "best_label", None)
                is_enrolled = true_label in enrolled
                accepted = sim >= float(t)
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
            far = fp / (fp + tn) if (fp + tn) else 0.0
            frr = fn / (fn + tp) if (fn + tp) else 0.0
            fars.append(far)
            frrs.append(frr)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(thresholds, fars, label="FAR", color="#d62728")
        ax.plot(thresholds, frrs, label="FRR", color="#1f77b4")
        if threshold_used is not None:
            ax.axvline(float(threshold_used), color="black", linestyle="--", label=f"threshold={threshold_used:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Taxa")
        ax.set_title("FAR/FRR em funcao do threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        primary = output_dir / "far_frr_threshold.png"
        _save(fig, primary)
        created.append(primary)
        if aliases:
            for name in ("far_frr.png", "far_frr_vs_threshold.png", "curva_far_frr.png", "farfrr.png"):
                (output_dir / name).write_bytes(primary.read_bytes())
                created.append(output_dir / name)
    except Exception:
        pass

    return created


# Alias para compatibilidade com nomes alternativos.
plot_results = generate_plots
create_plots = generate_plots
make_plots = generate_plots


__all__ = ["generate_plots", "plot_results", "create_plots", "make_plots"]