from typing import TYPE_CHECKING, Final

import polars as pl

from taxonomy_classifier.data.files import write_text_atomic
from taxonomy_classifier.data.split import Split
from taxonomy_classifier.model.labels import LabelSpace
from taxonomy_classifier.training.data import LabeledSet, TrainingData, class_frequencies
from taxonomy_classifier.training.oversampling import training_frame

if TYPE_CHECKING:
    from pathlib import Path

    from taxonomy_classifier.data.build import DataLayout

LABEL_SPACE_FILENAME: Final = "label_space.json"

TRAIN_TOKENS_FILENAME: Final = "train_tokens.bin"

VALIDATION_TOKENS_FILENAME: Final = "val_tokens.bin"


def label_space_path(layout: DataLayout) -> Path:
    return layout.processed_dir / LABEL_SPACE_FILENAME


def load_training_data(
    layout: DataLayout, *, min_class_count: int, label_space: LabelSpace | None = None
) -> tuple[LabelSpace, TrainingData]:
    train = training_frame(layout.dataset_path, layout.synthetic_path)
    validation = pl.scan_parquet(layout.dataset_path).filter(pl.col("split") == Split.VAL.value)

    label_space = label_space or LabelSpace.from_frame(
        train.filter(~pl.col("synthetic")), min_count=min_class_count
    )
    write_text_atomic(label_space_path(layout), label_space.to_json())

    markers = (
        tuple(validation.select("marker").collect().get_column("marker").to_list())
        if "marker" in validation.collect_schema().names()
        else ()
    )

    data = TrainingData(
        train=LabeledSet.from_frame(
            train, label_space, cache=layout.interim_dir / TRAIN_TOKENS_FILENAME
        ),
        frequencies=class_frequencies(train),
        validation=LabeledSet.from_frame(
            validation, label_space, cache=layout.interim_dir / VALIDATION_TOKENS_FILENAME
        ),
        validation_markers=markers,
    )

    return label_space, data
