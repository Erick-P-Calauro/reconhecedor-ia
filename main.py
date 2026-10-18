#!/usr/bin/env python3
"""Entrypoint do reconhecedor-ia.

Uso:
    python main.py build-demo              # monta o dataset demo em data/demo
    python main.py run                     # roda o experimento baseline
    python main.py run --dataset data/demo --name meu-experimento
    python main.py metrics                 # consulta o historico de metricas
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def cmd_build_demo(args: argparse.Namespace) -> int:
    from reconhecedor.demo import build_demo_dataset

    build_demo_dataset(args.dest)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from reconhecedor.dataloader import FolderDataloader
    from reconhecedor.evaluator import Evaluator
    from reconhecedor.face_detector import YuNetFaceDetector
    from reconhecedor.face_recognition import FaceRecognition
    from reconhecedor.model import BaselineHogModel
    from reconhecedor.pipeline import ExperimentConfig, run_experiment
    from reconhecedor.preprocessor import StandardPreProcessor

    config = ExperimentConfig(
        name=args.name,
        dataset_root=args.dataset,
        output_dir=args.output,
        threshold=args.threshold,
        crop_padding=args.padding,
    )
    detector_kwargs = {}
    if args.detector_model:
        detector_kwargs["model_path"] = args.detector_model

    result = run_experiment(
        config,
        gallery_loader=FolderDataloader(config.gallery_root),
        probe_loader=FolderDataloader(config.probe_root),
        detector=YuNetFaceDetector(**detector_kwargs),
        preprocessor=StandardPreProcessor(size=(args.image_size, args.image_size)),
        model=BaselineHogModel(),
        recognition=FaceRecognition(threshold=config.threshold),
        evaluator=Evaluator(),
    )

    metrics = result.metrics
    print()
    print(f"Experimento '{config.name}' concluido ({config.dataset_root})")
    print(f"  galeria: {result.n_gallery} imagens | probe: {result.n_probe} imagens")
    print(f"  descartadas sem face: {result.discarded_gallery + result.discarded_probe}")
    print(
        "  acuracia={accuracy:.4f} precisao={precision:.4f} recall={recall:.4f} "
        "FAR={far:.4f} FRR={frr:.4f}".format(
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            far=metrics.far,
            frr=metrics.frr,
        )
    )
    print(f"  metricas salvas em: {result.run_dir}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    from reconhecedor.evaluator.evaluator import HISTORY_FILENAME

    history_path = Path(args.output) / HISTORY_FILENAME
    if not history_path.is_file():
        print(f"Nenhuma metrica registrada ainda (arquivo inexistente: {history_path})")
        print("Rode 'python main.py run' primeiro.")
        return 1

    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"Historico vazio: {history_path}")
        return 1

    headers = ["timestamp", "experiment", "model", "accuracy", "precision", "recall", "far", "frr"]
    widths = {header: max(len(header), *(len(row.get(header, "")) for row in rows)) for header in headers}
    print("  ".join(header.ljust(widths[header]) for header in headers))
    for row in rows:
        print("  ".join(row.get(header, "").ljust(widths[header]) for header in headers))
    print(f"\nHistorico completo: {history_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline de reconhecimento facial replicavel (spec 001-004)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_demo = subparsers.add_parser(
        "build-demo", help="monta o dataset demo com imagens publicas em data/demo"
    )
    build_demo.add_argument("--dest", default="data/demo", help="pasta de destino (padrao: data/demo)")
    build_demo.set_defaults(func=cmd_build_demo)

    run = subparsers.add_parser("run", help="executa um experimento ponta a ponta")
    run.add_argument("--dataset", default="data/demo", help="raiz do dataset (com gallery/ e probe/)")
    run.add_argument("--name", default="baseline", help="nome do experimento (ADR 002)")
    run.add_argument("--output", default="results", help="pasta de saida das metricas")
    run.add_argument("--threshold", type=float, default=0.8, help="limiar de similaridade [0,1]")
    run.add_argument("--padding", type=float, default=0.25, help="margem do recorte da face")
    run.add_argument("--image-size", type=int, default=128, help="lado do quadrado pre-processado")
    run.add_argument("--detector-model", default=None, help="caminho alternativo do modelo do detector")
    run.set_defaults(func=cmd_run)

    metrics = subparsers.add_parser("metrics", help="lista o historico de metricas salvas")
    metrics.add_argument("--output", default="results", help="pasta de saida das metricas")
    metrics.set_defaults(func=cmd_metrics)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
