import argparse
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Final

import httpx
import torch

from taxonomy_classifier.data.build import BuildConfig, DataLayout, build_dataset
from taxonomy_classifier.data.download import download_all
from taxonomy_classifier.data.entrez import fetch_query
from taxonomy_classifier.data.sources.base import QueriedBundle, expected_filenames
from taxonomy_classifier.data.sources.registry import DEFAULT_SOURCES, DEFAULT_TAXONOMIES
from taxonomy_classifier.exceptions import DataError, GeneflowError, TrainingDeadlineError
from taxonomy_classifier.model.labels import LabelSpace
from taxonomy_classifier.training.final import FinalConfig, load_search_choice, train_final
from taxonomy_classifier.training.oversampling import OversamplingConfig, oversample_train
from taxonomy_classifier.training.progress import ProgressBarMonitor
from taxonomy_classifier.training.report import OVERALL, format_group_summary, format_report
from taxonomy_classifier.training.search import SUGGESTERS, SearchConfig, run_search
from taxonomy_classifier.training.setup import LABEL_SPACE_FILENAME, load_training_data
from taxonomy_classifier.training.trainer import LoggingMonitor, Precision

if TYPE_CHECKING:
    from collections.abc import Sequence

    import optuna

    from taxonomy_classifier.data.sources.base import DataSource, RemoteBundle, TaxonomySource

_LOGGER: Final = logging.getLogger(__name__)

_LOG_FORMAT: Final = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

_TIMEOUT: Final = httpx.Timeout(30.0, read=300.0)

_COMMANDS: Final = {
    "download": "Download and verify the raw files of every source",
    "build": "Build the harmonized dataset from already downloaded files",
    "augment": "Oversample rare classes of the training split with synthetic variants",
    "prepare": "Download, build and augment in one step",
    "cache": "Write the label space and the encoded sequence caches used for training",
    "search": "Run a hyperparameter search for one architecture",
    "train": "Train one configuration chosen by a hyperparameter search",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geneflow",
        description="Hierarchical taxonomic classification of DNA sequences.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in _COMMANDS.items():
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--data-dir", type=Path, default=Path("data"))

        if name in {"cache", "search", "train"}:
            command.add_argument("--min-class-count", type=int, default=3)

        if name in {"search", "train"}:
            _add_device_arguments(command)

        if name == "search":
            _add_search_arguments(command)

        if name == "train":
            _add_train_arguments(command)

    return parser


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    args = build_parser().parse_args(argv)

    command: str = args.command
    layout = DataLayout(root=args.data_dir)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format=_LOG_FORMAT)

    try:
        _run(args, DEFAULT_SOURCES, DEFAULT_TAXONOMIES, layout, transport=transport)

    except GeneflowError:
        _LOGGER.exception("Command %r failed", command)

        return 1

    return 0


def _add_search_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--architecture", choices=sorted(SUGGESTERS), required=True)
    command.add_argument("--storage", type=Path, default=None)
    command.add_argument("--trials", type=int, default=None)
    command.add_argument("--hours", type=float, default=None)
    command.add_argument("--epochs", type=int, default=6)
    command.add_argument("--samples-per-epoch", type=int, default=100_000)
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--validation-samples", type=int, default=20_000)


def _add_train_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--params", type=Path, required=True)
    command.add_argument("--trial", type=int, default=None)
    command.add_argument("--output", type=Path, default=None)
    command.add_argument("--hours", type=float, default=None)
    command.add_argument("--epochs", type=int, default=10)
    command.add_argument("--samples-per-epoch", type=int, default=None)
    command.add_argument("--validation-samples", type=int, default=50_000)
    command.add_argument("--report-samples", type=int, default=None)
    command.add_argument("--batch-size", type=int, default=128)
    command.add_argument("--log-every", type=int, default=500)
    command.add_argument("--refresh-seconds", type=float, default=0.5)
    command.add_argument("--resume", action="store_true")


def _add_device_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--precision", choices=[item.value for item in Precision], default="bf16")
    command.add_argument("--gpu-memory-fraction", type=float, default=None)
    command.add_argument("--seed", type=int, default=20260923)


def _run(
    args: argparse.Namespace,
    sources: Sequence[DataSource],
    taxonomies: Sequence[TaxonomySource],
    layout: DataLayout,
    *,
    transport: httpx.BaseTransport | None,
) -> None:
    command: str = args.command
    bundles: list[RemoteBundle] = [*taxonomies, *sources]

    if command in {"download", "prepare"}:
        with httpx.Client(transport=transport, timeout=_TIMEOUT, follow_redirects=True) as client:
            for bundle in bundles:
                raw_dir = layout.raw_dir(bundle)

                download_all(bundle.files, raw_dir, client=client)

                for query in bundle.queries if isinstance(bundle, QueriedBundle) else ():
                    fetch_query(query, raw_dir, client=client)

    if command in {"build", "prepare"}:
        _require_raw_files(bundles, layout)

        build_dataset(sources, layout, config=BuildConfig(), taxonomies=taxonomies)

    if command in {"augment", "prepare"}:
        _require_dataset(layout)

        oversample_train(
            layout.dataset_path,
            layout.synthetic_path,
            layout.augment_report_path,
            config=OversamplingConfig(),
        )

    if command == "cache":
        _require_training_inputs(layout)

        label_space, data = load_training_data(layout, min_class_count=args.min_class_count)
        _LOGGER.info(
            "Cached %d training and %d validation sequences; classes per level: %s",
            len(data.train),
            len(data.validation),
            label_space.sizes,
        )

    if command == "search":
        _search(args, layout)

    if command == "train":
        _train(args, layout)


def _search(args: argparse.Namespace, layout: DataLayout) -> None:
    _require_training_inputs(layout)

    try:
        config = SearchConfig(
            trials=args.trials,
            time_budget_hours=args.hours,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size,
            validation_samples=args.validation_samples,
            precision=Precision(args.precision),
            seed=args.seed,
        )

    except ValueError as error:
        raise DataError(str(error)) from error

    architecture: str = args.architecture
    storage: Path = args.storage or Path("checkpoints") / "search" / f"{architecture.lower()}.log"
    device = _device(args)

    label_space, data = load_training_data(layout, min_class_count=args.min_class_count)

    def monitor(trial: optuna.Trial) -> LoggingMonitor:
        _LOGGER.info("%s trial %d: %s", architecture, trial.number, trial.params)

        return LoggingMonitor(f"{architecture} trial {trial.number}")

    study = run_search(
        architecture,
        data,
        label_space,
        config=config,
        storage=storage,
        device=device,
        monitor_factory=monitor,
    )

    completed = [trial for trial in study.trials if trial.value is not None]

    if completed:
        _LOGGER.info(
            "%s best trial %d: %.4f %s",
            architecture,
            study.best_trial.number,
            study.best_value,
            study.best_params,
        )

    else:
        _LOGGER.warning("%s search finished without completed trials", architecture)


def _train(args: argparse.Namespace, layout: DataLayout) -> None:
    _require_training_inputs(layout)

    try:
        config = FinalConfig(
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            validation_samples=args.validation_samples,
            report_samples=args.report_samples,
            batch_size=args.batch_size,
            precision=Precision(args.precision),
            time_budget_hours=args.hours,
            log_every=args.log_every,
            seed=args.seed,
        )

    except ValueError as error:
        raise DataError(str(error)) from error

    choice = load_search_choice(args.params, trial=args.trial)
    name = f"{choice.architecture.lower()}-trial{choice.trial}"
    output: Path = args.output or Path("checkpoints") / name
    device = _device(args)
    resume: bool = args.resume
    saved_space = output / LABEL_SPACE_FILENAME
    fixed_space = (
        LabelSpace.from_json(saved_space.read_text(encoding="utf-8"))
        if resume and saved_space.exists()
        else None
    )

    if fixed_space is not None:
        _LOGGER.info("Resuming with the label space saved in %s", saved_space)

    label_space, data = load_training_data(
        layout, min_class_count=args.min_class_count, label_space=fixed_space
    )

    _LOGGER.info("Training %s from %s: %s", name, args.params, choice.params)

    steps_path = output / "steps.json"
    previous_steps = (
        json.loads(steps_path.read_text(encoding="utf-8")) if resume and steps_path.exists() else []
    )

    monitor = ProgressBarMonitor(
        sys.stdout,
        refresh_seconds=args.refresh_seconds,
        steps_path=steps_path,
        records=previous_steps,
    )

    try:
        result = train_final(
            choice,
            data,
            label_space,
            config=config,
            output_dir=output,
            device=device,
            monitor=monitor,
            resume=resume,
        )

    except TrainingDeadlineError:
        sys.stdout.write("\n")
        _LOGGER.warning("%s stopped by its time budget; checkpoints kept in %s", name, output)

        return

    if result.stopped:
        sys.stdout.write("\n")
        _LOGGER.warning(
            "%s stopped by its time budget after %d epochs; reporting its best checkpoint",
            name,
            len(result.history),
        )

    best = min(result.history, key=lambda record: record.validation.loss)

    sys.stdout.write(
        f"Classification report on validation, best epoch {best.epoch}/{len(result.history)}\n\n"
        f"{format_report(result.reports)}\n\n"
    )

    if result.marker_reports:
        grouped = {OVERALL: result.reports, **result.marker_reports}

        sys.stdout.write(
            f"Accuracy by marker\n\n{format_group_summary(grouped)}\n\n"
            f"Macro-F1 by marker\n\n{format_group_summary(grouped, metric='macro_f1')}\n\n"
        )

    sys.stdout.flush()

    _LOGGER.info("%s finished; checkpoints and reports in %s", name, output)


def _device(args: argparse.Namespace) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gpu_memory_fraction is not None and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)

    return device


def _require_raw_files(sources: Sequence[RemoteBundle], layout: DataLayout) -> None:
    missing = [
        str(layout.raw_dir(source) / filename)
        for source in sources
        for filename in expected_filenames(source)
        if not (layout.raw_dir(source) / filename).exists()
    ]

    if missing:
        msg = f"Missing raw files {missing}; run 'geneflow download' first"
        raise DataError(msg)


def _require_dataset(layout: DataLayout) -> None:
    if not layout.dataset_path.exists():
        msg = f"Missing dataset {layout.dataset_path}; run 'geneflow build' first"
        raise DataError(msg)


def _require_training_inputs(layout: DataLayout) -> None:
    _require_dataset(layout)

    if not layout.synthetic_path.exists():
        msg = f"Missing synthetic data {layout.synthetic_path}; run 'geneflow augment' first"
        raise DataError(msg)
