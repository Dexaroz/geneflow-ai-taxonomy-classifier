from dataclasses import dataclass
import json
from typing import TYPE_CHECKING, Any, Final

import polars as pl

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.model.classifier import build_classifier
from taxonomy_classifier.model.labels import LabelSpace
from taxonomy_classifier.training.data import LabeledSet
from taxonomy_classifier.training.final import (
    RUN_FILENAME,
    FinalConfig,
    SearchChoice,
    final_setup,
)
from taxonomy_classifier.training.report import OVERALL, classification_report
from taxonomy_classifier.training.setup import LABEL_SPACE_FILENAME
from taxonomy_classifier.training.trainer import BEST_CHECKPOINT, Precision, load_weights

if TYPE_CHECKING:
    from pathlib import Path

    import torch

    from taxonomy_classifier.data.build import DataLayout
    from taxonomy_classifier.data.split import Split
    from taxonomy_classifier.model.classifier import TaxonomyClassifier
    from taxonomy_classifier.training.report import LevelReport

_MARKER_COLUMN: Final = "marker"


@dataclass(frozen=True, slots=True)
class TrainedModel:
    model: TaxonomyClassifier
    label_space: LabelSpace
    epoch: int
    max_length: int


@dataclass(frozen=True, slots=True)
class SplitEvaluation:
    split: str
    epoch: int
    reports: list[LevelReport]
    marker_reports: dict[str, list[LevelReport]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "epoch": self.epoch,
            "overall": [report.to_dict() for report in self.reports],
            "markers": {
                marker: [report.to_dict() for report in levels]
                for marker, levels in self.marker_reports.items()
            },
        }


def evaluation_path(model_dir: Path, split: Split) -> Path:
    return model_dir / f"evaluation_{split.value}.json"


def load_trained_model(model_dir: Path, *, device: torch.device) -> TrainedModel:
    run = json.loads((model_dir / RUN_FILENAME).read_text(encoding="utf-8"))
    label_space = LabelSpace.from_json(
        (model_dir / LABEL_SPACE_FILENAME).read_text(encoding="utf-8")
    )
    setup = final_setup(
        SearchChoice(trial=int(run["search_trial"]), params=dict(run["params"])), FinalConfig()
    )
    model = build_classifier(setup.encoder, label_space, setup.heads).to(device)
    state = load_weights(model, model_dir / BEST_CHECKPOINT, device=device)

    return TrainedModel(
        model=model,
        label_space=label_space,
        epoch=int(state["epoch"]),
        max_length=setup.training.max_length,
    )


def split_data(
    layout: DataLayout, split: Split, label_space: LabelSpace
) -> tuple[LabeledSet, tuple[str, ...]]:
    frame = pl.scan_parquet(layout.dataset_path).filter(pl.col("split") == split.value)
    markers = tuple(frame.select(_MARKER_COLUMN).collect().get_column(_MARKER_COLUMN).to_list())
    data = LabeledSet.from_frame(
        frame, label_space, cache=layout.interim_dir / f"{split.value}_tokens.bin"
    )

    return data, markers


def evaluate_split(
    model_dir: Path,
    layout: DataLayout,
    split: Split,
    *,
    device: torch.device,
    batch_size: int,
    precision: Precision = Precision.BF16,
) -> SplitEvaluation:
    trained = load_trained_model(model_dir, device=device)
    data, markers = split_data(layout, split, trained.label_space)

    grouped = classification_report(
        trained.model,
        data,
        trained.label_space,
        batch_size=batch_size,
        device=device,
        precision=precision,
        max_length=trained.max_length,
        groups=markers,
    )
    reports = grouped.pop(OVERALL)

    result = SplitEvaluation(
        split=split.value, epoch=trained.epoch, reports=reports, marker_reports=grouped
    )
    write_text_atomic(
        evaluation_path(model_dir, split), json.dumps(result.to_dict(), indent=2) + "\n"
    )

    return result
