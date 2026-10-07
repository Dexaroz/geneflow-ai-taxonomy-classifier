from dataclasses import replace
from itertools import pairwise
import json
import math
from typing import TYPE_CHECKING, override

import polars as pl
import pytest
import torch
from torch import nn

from taxonomy_classifier.exceptions import TrainingResumeError
from taxonomy_classifier.model.classifier import build_classifier
from taxonomy_classifier.model.encoders import CnnConfig
from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS, LabelSpace
from taxonomy_classifier.training import trainer
from taxonomy_classifier.training.data import CropConfig, LabeledSet, SequenceBank, TrainingData
from taxonomy_classifier.training.trainer import (
    BEST_CHECKPOINT,
    HISTORY_FILENAME,
    LAST_CHECKPOINT,
    STATE_FILENAME,
    EpochRecord,
    Evaluation,
    LoggingMonitor,
    Precision,
    TrainingConfig,
    evaluate,
    learning_rate_factor,
    load_history,
    load_weights,
    train_classifier,
)

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.model.classifier import TaxonomyClassifier

CPU = torch.device("cpu")

SPACE = LabelSpace(classes=dict.fromkeys(LEVELS, ("a", "b")), min_count=1)

TINY_CNN = CnnConfig(embedding_dim=8, channels=(16,), kernel_size=5, dilations=(1, 2), dropout=0.0)


def _labeled(sequences: list[str], labels: list[int]) -> LabeledSet:
    series = pl.Series(sequences)
    targets = torch.tensor([[label] * len(LEVELS) for label in labels])

    return LabeledSet(
        bank=SequenceBank.from_batches([series], series.str.len_bytes()), targets=targets
    )


TRAIN = _labeled(["A" * 40, "C" * 40, "AAAAT" * 8, "CCCCG" * 8] * 4, [0, 1, 0, 1] * 4)

VALIDATION = _labeled(["A" * 30, "C" * 30], [0, 1])

DATA = TrainingData(
    train=TRAIN, frequencies=torch.ones(len(TRAIN), dtype=torch.double), validation=VALIDATION
)

CONFIG = TrainingConfig(
    epochs=4,
    batch_size=8,
    learning_rate=3e-3,
    warmup_fraction=0.1,
    crop=CropConfig(probability=0.0),
    evaluation_batch_size=2,
    precision=Precision.FP32,
    log_every=1,
)


class _RecordingMonitor:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []
        self.records: list[EpochRecord] = []

    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        self.events.append(("epoch", epoch, epochs, steps))

    def step_finished(self, step: int) -> None:
        self.events.append(("step", step))

    def metrics_updated(
        self, step: int, loss: float, sequences_per_second: float, learning_rate: float
    ) -> None:
        assert loss > 0.0
        assert sequences_per_second > 0.0
        assert learning_rate >= 0.0

        self.events.append(("metrics", step))

    def evaluation_started(self, batches: int) -> None:
        self.events.append(("evaluation", batches))

    def evaluation_step(self) -> None:
        self.events.append(("evaluation_step",))

    def epoch_finished(self, record: EpochRecord) -> None:
        self.records.append(record)


class _FirstClassModel(nn.Module):
    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        del mask

        return [torch.tensor([[5.0, 0.0]]).expand(len(tokens), 2) for _ in LEVELS]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
        ({"max_length": 0}, "max_length must be positive"),
        ({"length_bucket_batches": 0}, "length_bucket_batches must be positive"),
        ({"samples_per_epoch": 0}, "samples_per_epoch must be positive"),
        ({"learning_rate": 0.0}, "learning_rate and gradient_clip"),
        ({"weight_decay": -1.0}, "weight_decay non-negative"),
        ({"warmup_fraction": 1.0}, r"warmup_fraction must be in \[0, 1\)"),
    ],
)
def test_training_config_rejects_invalid_values(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    factors = [learning_rate_factor(step, total_steps=100, warmup_steps=10) for step in range(101)]

    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0)
    assert factors[55] == pytest.approx(0.5)
    assert factors[100] == pytest.approx(0.0, abs=1e-12)
    assert all(later <= earlier for earlier, later in pairwise(factors[9:]))


def test_evaluate_reports_accuracy_on_labeled_targets_only() -> None:
    data = _labeled(["ACGT", "ACGT", "ACGT"], [0, 1, 0])
    data.targets[2, :] = IGNORE_INDEX
    data.targets[:, 0] = IGNORE_INDEX

    model: TaxonomyClassifier = _FirstClassModel()  # type: ignore[assignment]

    result = evaluate(model, data, batch_size=2, device=CPU)

    assert math.isnan(result.accuracy["domain"])
    assert result.accuracy["genus"] == pytest.approx(0.5)
    assert result.loss > 0.0


def test_training_learns_a_separable_task_and_writes_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = build_classifier(TINY_CNN, SPACE)
    monitor = _RecordingMonitor()

    history = train_classifier(
        model,
        DATA,
        config=CONFIG,
        checkpoint_dir=tmp_path,
        device=CPU,
        metadata={"encoder": "cnn"},
        monitor=monitor,
    )

    assert [record.epoch for record in history] == [1, 2, 3, 4]
    assert monitor.records == history
    assert history[-1].train_loss < history[0].train_loss
    assert history[-1].validation.accuracy["genus"] == 1.0
    assert history[-1].learning_rate == pytest.approx(0.0, abs=1e-9)
    assert (tmp_path / BEST_CHECKPOINT).exists()
    assert (tmp_path / LAST_CHECKPOINT).exists()

    written = json.loads((tmp_path / HISTORY_FILENAME).read_text(encoding="utf-8"))

    assert written == [record.to_dict() for record in history]
    assert set(history[0].train_accuracy) == set(LEVELS)
    assert load_history(tmp_path) == written


def test_saved_weights_restore_the_trained_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    trained = build_classifier(TINY_CNN, SPACE)
    train_classifier(trained, DATA, config=CONFIG, checkpoint_dir=tmp_path, device=CPU)

    restored = build_classifier(TINY_CNN, SPACE)
    state = load_weights(restored, tmp_path / LAST_CHECKPOINT, device=CPU)

    assert state["epoch"] == CONFIG.epochs
    assert state["config"]["batch_size"] == CONFIG.batch_size

    for original, loaded in zip(
        trained.state_dict().values(), restored.state_dict().values(), strict=True
    ):
        assert torch.equal(original, loaded)


def test_monitor_follows_every_step_and_evaluation(tmp_path: Path) -> None:
    config = TrainingConfig(
        epochs=1,
        samples_per_epoch=12,
        batch_size=4,
        crop=CropConfig(probability=0.0),
        evaluation_batch_size=1,
        precision=Precision.FP32,
        log_every=2,
    )
    monitor = _RecordingMonitor()

    train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=config,
        checkpoint_dir=tmp_path,
        device=CPU,
        monitor=monitor,
    )

    assert monitor.events == [
        ("epoch", 1, 1, 3),
        ("step", 1),
        ("step", 2),
        ("metrics", 2),
        ("step", 3),
        ("evaluation", 2),
        ("evaluation_step",),
        ("evaluation_step",),
    ]
    assert len(monitor.records) == 1


def test_load_history_without_a_run_is_empty(tmp_path: Path) -> None:
    assert load_history(tmp_path) == []


def test_best_checkpoint_keeps_the_lowest_validation_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    losses = iter([1.0, 2.0])

    def worsening(*_: object, **__: object) -> Evaluation:
        return Evaluation(loss=next(losses), accuracy=dict.fromkeys(LEVELS, 0.0))

    monkeypatch.setattr(trainer, "evaluate", worsening)
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        crop=CropConfig(probability=0.0),
        precision=Precision.FP32,
        log_every=1000,
    )

    train_classifier(
        build_classifier(TINY_CNN, SPACE), DATA, config=config, checkpoint_dir=tmp_path, device=CPU
    )

    best = torch.load(tmp_path / BEST_CHECKPOINT, weights_only=True)
    last = torch.load(tmp_path / LAST_CHECKPOINT, weights_only=True)

    assert (best["epoch"], last["epoch"]) == (1, 2)


def test_float16_training_uses_loss_scaling(tmp_path: Path) -> None:
    torch.manual_seed(0)
    config = TrainingConfig(
        epochs=1,
        batch_size=8,
        crop=CropConfig(probability=0.0),
        precision=Precision.FP16,
        log_every=1000,
    )

    history = train_classifier(
        build_classifier(TINY_CNN, SPACE), DATA, config=config, checkpoint_dir=tmp_path, device=CPU
    )

    state = torch.load(tmp_path / LAST_CHECKPOINT, weights_only=True)

    assert math.isfinite(history[0].train_loss)
    assert state["config"]["precision"] == "fp16"


def test_logging_monitor_reports_progress(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    monitor = LoggingMonitor("CNN trial 3")
    record = EpochRecord(
        epoch=1,
        train_loss=1.0,
        validation=Evaluation(loss=0.5, accuracy=dict.fromkeys(LEVELS, 0.25)),
        learning_rate=1e-3,
        seconds=90.0,
    )

    monitor.epoch_started(1, 2, 10)
    monitor.step_finished(1)
    monitor.metrics_updated(5, 1.2, 800.0, 1e-3)
    monitor.evaluation_started(3)
    monitor.evaluation_step()
    monitor.epoch_finished(record)

    assert [message.split(":")[0] for message in caplog.messages] == ["CNN trial 3"] * 3
    assert "800 sequences/s" in caplog.messages[1]
    assert "genus 0.250" in caplog.messages[2]


class _InterruptedError(Exception):
    pass


class _StopBeforeEpoch(_RecordingMonitor):
    def __init__(self, epoch: int) -> None:
        super().__init__()
        self.stop_at = epoch

    @override
    def epoch_started(self, epoch: int, epochs: int, steps: int) -> None:
        if epoch == self.stop_at:
            raise _InterruptedError

        super().epoch_started(epoch, epochs, steps)


def _parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]


def test_resuming_after_an_interruption_matches_an_uninterrupted_run(tmp_path: Path) -> None:
    torch.manual_seed(0)
    straight = build_classifier(TINY_CNN, SPACE)
    expected = train_classifier(
        straight, DATA, config=CONFIG, checkpoint_dir=tmp_path / "straight", device=CPU
    )

    torch.manual_seed(0)
    interrupted = build_classifier(TINY_CNN, SPACE)

    with pytest.raises(_InterruptedError):
        train_classifier(
            interrupted,
            DATA,
            config=CONFIG,
            checkpoint_dir=tmp_path / "resumed",
            device=CPU,
            monitor=_StopBeforeEpoch(3),
        )

    torch.manual_seed(123)
    resumed = build_classifier(TINY_CNN, SPACE)
    monitor = _RecordingMonitor()
    history = train_classifier(
        resumed,
        DATA,
        config=CONFIG,
        checkpoint_dir=tmp_path / "resumed",
        device=CPU,
        monitor=monitor,
        resume=True,
    )

    assert [record.epoch for record in monitor.records] == [3, 4]
    assert [record.to_dict()["val_loss"] for record in history] == pytest.approx(
        [record.to_dict()["val_loss"] for record in expected]
    )

    for left, right in zip(_parameters(straight), _parameters(resumed), strict=True):
        torch.testing.assert_close(left, right)


def test_resuming_a_finished_run_trains_nothing(tmp_path: Path) -> None:
    train_classifier(
        build_classifier(TINY_CNN, SPACE), DATA, config=CONFIG, checkpoint_dir=tmp_path, device=CPU
    )
    monitor = _RecordingMonitor()

    history = train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=CONFIG,
        checkpoint_dir=tmp_path,
        device=CPU,
        monitor=monitor,
        resume=True,
    )

    assert len(history) == CONFIG.epochs
    assert monitor.records == []


def test_resume_without_a_saved_state_starts_from_scratch(tmp_path: Path) -> None:
    history = train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=replace(CONFIG, epochs=1),
        checkpoint_dir=tmp_path,
        device=CPU,
        resume=True,
    )

    assert [record.epoch for record in history] == [1]
    assert (tmp_path / STATE_FILENAME).exists()


def test_resume_extends_the_schedule_to_more_epochs(tmp_path: Path) -> None:
    short = train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=replace(CONFIG, epochs=2),
        checkpoint_dir=tmp_path,
        device=CPU,
    )
    monitor = _RecordingMonitor()

    history = train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=replace(CONFIG, epochs=5),
        checkpoint_dir=tmp_path,
        device=CPU,
        monitor=monitor,
        resume=True,
    )

    assert [record.epoch for record in monitor.records] == [3, 4, 5]
    assert [record.epoch for record in history] == [1, 2, 3, 4, 5]
    assert history[2].learning_rate > short[-1].learning_rate
    assert history[-1].learning_rate < history[2].learning_rate


def test_resume_refuses_a_shorter_schedule(tmp_path: Path) -> None:
    train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=replace(CONFIG, epochs=3),
        checkpoint_dir=tmp_path,
        device=CPU,
    )

    with pytest.raises(TrainingResumeError, match="spans 3 epochs"):
        train_classifier(
            build_classifier(TINY_CNN, SPACE),
            DATA,
            config=replace(CONFIG, epochs=2),
            checkpoint_dir=tmp_path,
            device=CPU,
            resume=True,
        )


def test_resume_refuses_a_different_configuration(tmp_path: Path) -> None:
    train_classifier(
        build_classifier(TINY_CNN, SPACE),
        DATA,
        config=replace(CONFIG, epochs=1),
        checkpoint_dir=tmp_path,
        device=CPU,
    )

    with pytest.raises(TrainingResumeError, match="another configuration"):
        train_classifier(
            build_classifier(TINY_CNN, SPACE),
            DATA,
            config=replace(CONFIG, epochs=2, learning_rate=1e-3),
            checkpoint_dir=tmp_path,
            device=CPU,
            resume=True,
        )
