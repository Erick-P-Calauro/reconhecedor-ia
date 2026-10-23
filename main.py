from __future__ import annotations
import argparse
import csv
import importlib
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


def _load_experiment(requested: str | None):
    """Importa ``experiments.<nome>.experiment`` (descoberta ADR 002).

    Retorna ``(modulo, nomes_descobertos)``; o modulo e ``None`` quando o
    experimento pedido nao existe (mensagem ja impressa).
    """
    from experiments import discover_experiments

    names = discover_experiments()
    if not names:
        print("Nenhum experimento encontrado em src/experiments/ .")
        print("Crie uma subpasta dedicada com config.json e experiment.py.")
        return None, names

    name = requested
    if name is None:
        if len(names) == 1:
            name = names[0]
        else:
            print("Mais de um experimento disponivel; use --experiment <nome>.")
            print(f"Disponiveis: {', '.join(names)}")
            return None, names

    if name not in names:
        print(f"Experimento '{name}' nao encontrado em src/experiments/.")
        print(f"Disponiveis: {', '.join(names)}")
        return None, names

    module = importlib.import_module(f"experiments.{name}.experiment")
    return module, names


def cmd_experiments(args: argparse.Namespace) -> int:
    from experiments import discover_experiments, experiment_dir
    from experiments.config import load_config

    names = discover_experiments()
    if not names:
        print("Nenhum experimento encontrado em src/experiments/ (ADR 002).")
        return 1

    for name in names:
        config = load_config(experiment_dir(name))
        params = config.parameters
        print(config.name)
        print(f"  deteccao: {config.face_detection_model} | reconhecimento: {config.face_recognition_model}")
        print(
            "  dataset: {} | limiar: {} | recorte: {} | imagem: {}x{}".format(
                config.dataset,
                params.get("threshold"),
                params.get("crop_padding"),
                params.get("image_size"),
                params.get("image_size"),
            )
        )
        print(
            "  dataset_parameters: sample_size={} n_classes={} min_photos_per_class={} validation_ratio={} test_ratio={}".format(
                config.sample_size,
                config.n_classes,
                config.min_photos_per_class,
                config.validation_ratio,
                config.test_ratio,
            )
        )
        print(f"  resultados: {config.results_dir}")
        print()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    module, _ = _load_experiment(args.experiment)
    if module is None:
        return 1

    overrides = {
        key: value
        for key, value in {
            "dataset": args.dataset,
            "threshold": args.threshold,
            "crop_padding": args.padding,
            "image_size": args.image_size,
            "detector_model_path": args.detector_model,
            "rematerialize": args.rematerialize if getattr(args, "rematerialize", False) else None,
            "sample_size": args.sample_size,
            "n_classes": args.n_classes,
            "min_photos_per_class": args.min_photos_per_class,
            "validation_ratio": args.validation_ratio,
            "test_ratio": args.test_ratio,
        }.items()
        if value is not None
    }

    config = module.load()
    result = module.run(**overrides)

    metrics = result.metrics
    print()
    print(f"Experimento '{config.name}' concluido")
    print(f"  config: {module.CONFIG_PATH}")
    print(f"  galeria: {result.n_gallery} imagens | validacao: {result.n_validation} imagens | teste: {result.n_test} imagens")
    print(f"  descartadas sem face: {result.discarded_gallery + result.discarded_validation + result.discarded_test}")
    print(
        "  teste: acuracia={accuracy:.4f} precisao={precision:.4f} recall={recall:.4f} "
        "FAR={far:.4f} FRR={frr:.4f}".format(
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            far=metrics.far,
            frr=metrics.frr,
        )
    )
    if result.validation_metrics is not None:
        vm = result.validation_metrics
        print(
            "  validacao: acuracia={accuracy:.4f} precisao={precision:.4f} recall={recall:.4f} "
            "FAR={far:.4f} FRR={frr:.4f}".format(
                accuracy=vm.accuracy,
                precision=vm.precision,
                recall=vm.recall,
                far=vm.far,
                frr=vm.frr,
            )
        )
    print(f"  metricas salvas em: {result.run_dir}")
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    from experiments import discover_experiments, experiment_dir
    from experiments.config import load_config
    from reconhecedor.evaluator.evaluator import HISTORY_FILENAME

    names = discover_experiments()
    if args.experiment:
        if args.experiment not in names:
            available = ", ".join(names) if names else "(nenhum)"
            print(f"Experimento '{args.experiment}' nao encontrado. Disponiveis: {available}")
            return 1
        names = [args.experiment]

    headers = ["timestamp", "experiment", "model", "accuracy", "precision", "recall", "far", "frr"]
    printed_any = False
    for name in names:
        # ADR 002 (caracteristica 4): o historico vive na subpasta do experimento.
        config = load_config(experiment_dir(name))
        history_path = config.results_dir / HISTORY_FILENAME
        if not history_path.is_file():
            continue

        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"Historico vazio: {history_path}")
            continue

        printed_any = True
        print(f"Experimento '{name}' — {history_path}")
        widths = {header: max(len(header), *(len(row.get(header, "")) for row in rows)) for header in headers}
        print("  ".join(header.ljust(widths[header]) for header in headers))
        for row in rows:
            print("  ".join(row.get(header, "").ljust(widths[header]) for header in headers))
        print()

    if not printed_any:
        print(f"Nenhuma metrica registrada ainda (arquivo inexistente: src/experiments/*/results/{HISTORY_FILENAME})")
        print("Rode 'python main.py run' primeiro.")
        return 1
    return 0


class FriendlyArgumentParser(argparse.ArgumentParser):
    """ArgumentParser que anexa dica de ``help`` aos erros de parsing."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(
            2,
            f"{self.prog}: error: {message}\n"
            "Use 'python main.py help' para ver os comandos disponíveis.\n",
        )


def cmd_help(args: argparse.Namespace) -> int:
    parser = args._parser
    command_parsers = getattr(parser, "_command_parsers", {})
    command_help = getattr(parser, "_command_help", {})
    names = list(command_parsers)

    requested = getattr(args, "command_name", None)
    if requested is None:
        print("Comandos disponíveis:")
        for name in names:
            print(f"  {name}  {command_help.get(name, '')}")
        print()
        parser.print_help()
        return 0

    if requested not in command_parsers:
        print(f"Comando desconhecido: {requested}")
        print(f"Comandos válidos: {', '.join(names)}")
        return 1

    command_parsers[requested].print_help()
    return 0


def build_parser() -> FriendlyArgumentParser:
    parser = FriendlyArgumentParser(
        description="Pipeline de reconhecimento facial replicavel (spec 001-005, ADR 001-006)."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=FriendlyArgumentParser
    )

    experiments = subparsers.add_parser(
        "experiments",
        help="lista os experimentos de src/experiments/ (ADR 002)",
    )
    experiments.set_defaults(func=cmd_experiments)

    run = subparsers.add_parser(
        "run",
        help="executa um experimento de src/experiments/ ponta a ponta",
    )
    run.add_argument(
        "--experiment", "-e", default=None, help="experimento a executar (padrao: unico disponivel)"
    )
    run.add_argument("--dataset", default=None, help="sobrescreve o dataset da configuracao")
    run.add_argument("--threshold", type=float, default=None, help="sobrescreve o limiar de similaridade [0,1]")
    run.add_argument("--padding", type=float, default=None, help="sobrescreve a margem do recorte da face")
    run.add_argument("--image-size", type=int, default=None, help="sobrescreve o lado do quadrado pre-processado")
    run.add_argument("--detector-model", default=None, help="caminho alternativo do modelo do detector")
    run.add_argument("--rematerialize", action="store_true", help="re-materializa o dataset (ADR 004)")
    run.add_argument("--sample-size", type=int, default=None, help="tamanho da amostra (max imagens por classe)")
    run.add_argument("--n-classes", type=int, default=None, help="quantidade de classes")
    run.add_argument("--min-photos-per-class", type=int, default=None, help="fotos minimas por classe")
    run.add_argument("--validation-ratio", type=float, default=None, help="proporcao de validacao")
    run.add_argument("--test-ratio", type=float, default=None, help="proporcao de teste")
    run.set_defaults(func=cmd_run)

    metrics = subparsers.add_parser(
        "metrics",
        help="lista o historico de metricas de src/experiments/<nome>/results/",
    )
    metrics.add_argument("--experiment", "-e", default=None, help="restringe a um experimento")
    metrics.set_defaults(func=cmd_metrics)

    help_parser = subparsers.add_parser(
        "help",
        help="aglutina e explica todos os comandos disponíveis",
    )
    help_parser.add_argument(
        "command_name", nargs="?", default=None, help="comando para detalhar (opcional)"
    )
    help_parser.set_defaults(func=cmd_help, _parser=parser)

    parser._command_parsers = {
        "experiments": experiments,
        "run": run,
        "metrics": metrics,
        "help": help_parser,
    }
    parser._command_help = {
        "experiments": "lista os experimentos de src/experiments/ (ADR 002)",
        "run": "executa um experimento de src/experiments/ ponta a ponta",
        "metrics": "lista o historico de metricas de src/experiments/<nome>/results/",
        "help": "aglutina e explica todos os comandos disponíveis",
    }

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
