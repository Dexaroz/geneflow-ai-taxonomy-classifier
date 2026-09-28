from dataclasses import replace
import json
import logging
import runpy
import sys
import time
from typing import TYPE_CHECKING, override

import httpx
import optuna
import pytest

from taxonomy_classifier import cli
from taxonomy_classifier.data.build import DataLayout
from taxonomy_classifier.exceptions import TrainingDeadlineError
from taxonomy_classifier.training import setup
from taxonomy_classifier.training.progress import ProgressBarMonitor

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.sources.base import DataSource, TaxonomySource
    from taxonomy_classifier.data.sources.genbank import GenBankSource
    from taxonomy_classifier.model.labels import LabelSpace
    from taxonomy_classifier.training.data import TrainingData


@pytest.fixture
def transport(
    sources: tuple[DataSource, ...],
    taxonomies: tuple[TaxonomySource, ...],
    fixtures_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> httpx.MockTransport:
    monkeypatch.setattr(cli, "DEFAULT_SOURCES", sources)
    monkeypatch.setattr(cli, "DEFAULT_TAXONOMIES", taxonomies)

    def serve(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, content=(fixtures_dir / request.url.path.rsplit("/")[-1]).read_bytes()
        )

    return httpx.MockTransport(serve)


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["build"])

    assert args.command == "build"
    assert str(args.data_dir) == "data"
    assert args.verbose is False


@pytest.mark.parametrize("argv", [[], ["unknown"], ["build", "--release", "144"]])
def test_invalid_arguments_exit_with_usage_error(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)

    assert caught.value.code == 2


def test_build_without_raw_files_fails_cleanly(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exit_code = cli.main(["build", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'geneflow download' first" in caplog.text


def test_download_runs_entrez_queries(
    genbank_source: GenBankSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "DEFAULT_SOURCES", (genbank_source,))
    monkeypatch.setattr(cli, "DEFAULT_TAXONOMIES", ())
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(
                200, json={"esearchresult": {"count": "1", "webenv": "E", "querykey": "1"}}
            )

        return httpx.Response(200, text=">OR1.1 Micromonas pusilla rbcL\nACGT\n")

    layout = DataLayout(root=tmp_path)
    transport = httpx.MockTransport(serve)

    assert cli.main(["download", "--data-dir", str(tmp_path)], transport=transport) == 0
    assert (layout.raw_dir(genbank_source) / genbank_source.query.filename).exists()


def test_download_fetches_every_source_file(
    sources: tuple[DataSource, ...],
    transport: httpx.MockTransport,
    tmp_path: Path,
) -> None:
    exit_code = cli.main(["download", "--data-dir", str(tmp_path)], transport=transport)

    layout = DataLayout(root=tmp_path)

    assert exit_code == 0

    for source in sources:
        for remote in source.files:
            assert (layout.raw_dir(source) / remote.filename).exists()

    assert not layout.processed_dir.exists()


def test_prepare_downloads_and_builds(transport: httpx.MockTransport, tmp_path: Path) -> None:
    exit_code = cli.main(["-v", "prepare", "--data-dir", str(tmp_path)], transport=transport)

    layout = DataLayout(root=tmp_path)
    report = json.loads(layout.report_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert layout.dataset_path.exists()
    assert layout.synthetic_path.exists()
    assert layout.augment_report_path.exists()
    assert report["taxonomies"] == {"ncbi_taxonomy_test": 10}
    assert [source["read"] for source in report["sources"]] == [7, 7, 5, 4]


def test_module_entrypoint_exits_with_command_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["geneflow", "build", "--data-dir", str(tmp_path)])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("taxonomy_classifier", run_name="__main__")

    assert caught.value.code == 1


def test_augment_without_dataset_fails_cleanly(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    exit_code = cli.main(["augment", "--data-dir", str(tmp_path)])

    assert exit_code == 1
    assert "run 'geneflow build' first" in caplog.text


def test_cache_writes_the_label_space_and_token_caches(built_layout: DataLayout) -> None:
    exit_code = cli.main(["cache", "--data-dir", str(built_layout.root), "--min-class-count", "1"])

    assert exit_code == 0
    assert (built_layout.processed_dir / "label_space.json").exists()
    assert (built_layout.interim_dir / "train_tokens.bin").exists()


def test_cache_requires_the_synthetic_data(
    built_layout: DataLayout, caplog: pytest.LogCaptureFixture
) -> None:
    built_layout.synthetic_path.unlink()

    exit_code = cli.main(["cache", "--data-dir", str(built_layout.root)])

    assert exit_code == 1
    assert "run 'geneflow augment' first" in caplog.text


def test_search_runs_and_writes_its_results(built_layout: DataLayout, tmp_path: Path) -> None:
    storage = tmp_path / "search" / "cnn.log"

    exit_code = cli.main(
        [
            "search",
            "--data-dir",
            str(built_layout.root),
            "--min-class-count",
            "1",
            "--architecture",
            "CNN",
            "--storage",
            str(storage),
            "--trials",
            "1",
            "--epochs",
            "1",
            "--samples-per-epoch",
            "4",
            "--batch-size",
            "2",
            "--validation-samples",
            "2",
            "--precision",
            "fp32",
            "--gpu-memory-fraction",
            "0.9",
        ]
    )

    assert exit_code == 0
    assert storage.exists()
    assert storage.with_name("cnn-trials.json").exists()


def test_search_without_any_budget_fails_cleanly(
    built_layout: DataLayout, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = cli.main(["search", "--data-dir", str(built_layout.root), "--architecture", "CNN"])

    assert exit_code == 1
    assert "needs a number of trials" in caplog.text


def test_search_warns_when_no_trial_completes(
    built_layout: DataLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cli, "run_search", lambda *_, **__: optuna.create_study())

    exit_code = cli.main(
        ["search", "--data-dir", str(built_layout.root), "--architecture", "CNN", "--trials", "1"]
    )

    assert exit_code == 0
    assert "finished without completed trials" in caplog.text


def test_search_reports_the_best_trial(
    built_layout: DataLayout, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=0.42, params={}, distributions={}))
    monkeypatch.setattr(cli, "run_search", lambda *_, **__: study)

    exit_code = cli.main(
        ["search", "--data-dir", str(built_layout.root), "--architecture", "CNN", "--trials", "1"]
    )

    assert exit_code == 0
    assert "best trial 0: 0.4200" in caplog.text


def _best_cnn(tmp_path: Path) -> Path:
    path = tmp_path / "cnn-best.json"
    params = {
        "dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.05,
        "warmup_fraction": 0.05,
        "sampling_power": 0.25,
        "crop_probability": 0.2,
        "cnn_width": "small",
        "cnn_blocks": "two",
        "cnn_kernel": 5,
    }
    path.write_text(json.dumps({"trial": 12, "value": 0.6, "params": params}), encoding="utf-8")

    return path


def _train_arguments(built_layout: DataLayout, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--data-dir",
        str(built_layout.root),
        "--min-class-count",
        "1",
        "--params",
        str(_best_cnn(tmp_path)),
        "--output",
        str(tmp_path / "final"),
        "--samples-per-epoch",
        "4",
        "--batch-size",
        "2",
        "--precision",
        "fp32",
        *extra,
    ]


def test_train_writes_checkpoints_progress_and_reports(
    built_layout: DataLayout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        _train_arguments(built_layout, tmp_path, "--epochs", "1", "--log-every", "1")
    )

    output = capsys.readouterr().out
    steps = json.loads((tmp_path / "final" / "steps.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert (tmp_path / "final" / "best.pt").exists()
    assert (tmp_path / "final" / "report.json").exists()
    assert "Epoch 1/1" in output
    assert "val_loss:" in output
    assert "Classification report on validation, best epoch 1/1" in output
    assert "macro-f1" in output
    assert "Accuracy by marker" in output
    assert "Macro-F1 by marker" in output
    assert (tmp_path / "final" / "report_by_marker.json").exists()
    assert steps[0]["step"] == 1


def test_train_skips_the_marker_tables_without_markers(
    built_layout: DataLayout,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load = setup.load_training_data

    def without_markers(
        layout: DataLayout, *, min_class_count: int, label_space: LabelSpace | None = None
    ) -> tuple[LabelSpace, TrainingData]:
        del label_space
        label_space, data = load(layout, min_class_count=min_class_count)

        return label_space, replace(data, validation_markers=())

    monkeypatch.setattr(cli, "load_training_data", without_markers)

    assert cli.main(_train_arguments(built_layout, tmp_path, "--epochs", "1")) == 0
    assert "by marker" not in capsys.readouterr().out
    assert not (tmp_path / "final" / "report_by_marker.json").exists()


def test_train_reports_the_best_epoch_when_the_budget_runs_out_later(
    built_layout: DataLayout,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterFirstEpoch(ProgressBarMonitor):
        @override
        def step_finished(self, step: int) -> None:
            super().step_finished(step)

            if self.records and self.records[-1]["epoch"] == 2:
                msg = "The time budget ran out before training finished"
                raise TrainingDeadlineError(msg)

    monkeypatch.setattr(cli, "ProgressBarMonitor", StopAfterFirstEpoch)

    exit_code = cli.main(
        _train_arguments(built_layout, tmp_path, "--epochs", "3", "--log-every", "1")
    )

    assert exit_code == 0
    assert "stopped by its time budget after 1 epochs" in caplog.text
    assert "best epoch 1/1" in capsys.readouterr().out


def test_train_resumes_from_its_output_directory(
    built_layout: DataLayout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _train_arguments(built_layout, tmp_path, "--epochs", "2", "--log-every", "1")

    assert cli.main([*first, "--resume"]) == 0

    steps = json.loads((tmp_path / "final" / "steps.json").read_text(encoding="utf-8"))
    capsys.readouterr()

    assert cli.main([*first, "--resume"]) == 0
    assert "Epoch" not in capsys.readouterr().out
    assert json.loads((tmp_path / "final" / "steps.json").read_text(encoding="utf-8")) == steps


def test_resuming_reuses_the_saved_label_space(
    built_layout: DataLayout, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    arguments = [*_train_arguments(built_layout, tmp_path, "--epochs", "1"), "--resume"]
    saved = tmp_path / "final" / "label_space.json"

    assert cli.main(arguments) == 0

    original = saved.read_text(encoding="utf-8")

    assert cli.main([*arguments[:-1], "--min-class-count", "2", "--resume"]) == 0
    assert "Resuming with the label space saved in" in caplog.text
    assert saved.read_text(encoding="utf-8") == original


def test_evaluate_reports_a_trained_model_on_a_split(
    built_layout: DataLayout, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_train_arguments(built_layout, tmp_path, "--epochs", "1")) == 0

    capsys.readouterr()
    model_dir = tmp_path / "final"
    arguments = ["evaluate", "--data-dir", str(built_layout.root), "--model-dir", str(model_dir)]

    assert cli.main([*arguments, "--split", "val", "--precision", "fp32"]) == 0

    output = capsys.readouterr().out
    written = json.loads((model_dir / "evaluation_val.json").read_text(encoding="utf-8"))

    assert "Classification report on val, epoch 1" in output
    assert "Accuracy by marker" in output
    assert written["split"] == "val"
    assert written["epoch"] == 1
    assert [level["level"] for level in written["overall"]][-1] == "species"
    assert set(written["markers"]) <= {"ssu", "coi", "its", "rbcl", "matk"}


def test_evaluate_rejects_a_directory_without_a_model(
    built_layout: DataLayout, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = cli.main(
        ["evaluate", "--data-dir", str(built_layout.root), "--model-dir", str(tmp_path)]
    )

    assert exit_code == 1
    assert "is not a trained model directory" in caplog.text


def test_train_stops_cleanly_at_its_time_budget(
    built_layout: DataLayout, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = cli.main(_train_arguments(built_layout, tmp_path, "--hours", "1e-9"))

    assert exit_code == 0
    assert "stopped by its time budget" in caplog.text


def test_train_rejects_an_invalid_configuration(
    built_layout: DataLayout, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    exit_code = cli.main(_train_arguments(built_layout, tmp_path, "--epochs", "0"))

    assert exit_code == 1
    assert "epochs, batch_size and log_every must be positive" in caplog.text
