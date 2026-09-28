import math
from typing import TYPE_CHECKING, cast, override

import polars as pl
import pytest
import torch

from taxonomy_classifier.model.labels import IGNORE_INDEX, LEVELS, LabelSpace
from taxonomy_classifier.training.data import LabeledSet, SequenceBank
from taxonomy_classifier.training.report import (
    OVERALL,
    LevelReport,
    classification_report,
    format_group_summary,
    format_report,
)

if TYPE_CHECKING:
    from taxonomy_classifier.model.classifier import TaxonomyClassifier

CPU = torch.device("cpu")

SPACE = LabelSpace(classes=dict.fromkeys(LEVELS, ("a", "b", "c")), min_count=1)

TARGETS = [0, 0, 1, 2]

PREDICTED = [0, 1, 1, 1]


class FixedModel(torch.nn.Module):
    @override
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
        del mask
        logits = torch.nn.functional.one_hot(torch.tensor(PREDICTED[: len(tokens)]), 3).float()

        return [logits] * len(LEVELS)


def _data() -> LabeledSet:
    series = pl.Series(["ACGT" * 5] * len(TARGETS))
    known = [[target] * (len(LEVELS) - 1) + [IGNORE_INDEX] for target in TARGETS]

    return LabeledSet(
        bank=SequenceBank.from_batches([series], series.str.len_bytes()),
        targets=torch.tensor(known),
    )


def _grouped(groups: tuple[str, ...] = (), batch_size: int = 8) -> dict[str, list[LevelReport]]:
    return classification_report(
        cast("TaxonomyClassifier", FixedModel()),
        _data(),
        SPACE,
        batch_size=batch_size,
        device=CPU,
        groups=groups,
    )


def _reports() -> list[LevelReport]:
    return _grouped()[OVERALL]


def test_report_matches_hand_computed_metrics() -> None:
    genus = _reports()[LEVELS.index("genus")]

    assert genus.accuracy == pytest.approx(0.5)
    assert genus.macro_precision == pytest.approx((1 + 1 / 3) / 3)
    assert genus.macro_recall == pytest.approx(0.5)
    assert genus.macro_f1 == pytest.approx((2 / 3 + 0.5) / 3)
    assert genus.weighted_precision == pytest.approx((2 + 1 / 3) / 4)
    assert genus.weighted_recall == pytest.approx(0.5)
    assert genus.weighted_f1 == pytest.approx((4 / 3 + 0.5) / 4)
    assert (genus.support, genus.classes) == (4, 3)


def test_levels_without_labels_report_no_metrics() -> None:
    species = _reports()[LEVELS.index("species")]

    assert math.isnan(species.accuracy)
    assert math.isnan(species.weighted_f1)
    assert (species.support, species.classes) == (0, 0)
    assert species.to_dict()["level"] == "species"


def test_format_report_lists_every_level() -> None:
    lines = format_report(_reports()).splitlines()

    assert "macro-f1" in lines[0]
    assert "support" in lines[0]
    assert [line.split()[0] for line in lines[2:]] == list(LEVELS)
    assert lines[2 + LEVELS.index("genus")].split()[1:2] == ["0.5000"]


def test_groups_get_their_own_reports_across_batches() -> None:
    grouped = _grouped(("coi", "ssu", "ssu", "ssu"), batch_size=3)
    genus = LEVELS.index("genus")

    assert list(grouped) == [OVERALL, "coi", "ssu"]
    assert grouped[OVERALL][genus].accuracy == pytest.approx(0.5)
    assert grouped["coi"][genus].accuracy == pytest.approx(1.0)
    assert grouped["ssu"][genus].accuracy == pytest.approx(1 / 3)
    assert (grouped["coi"][genus].support, grouped["ssu"][genus].support) == (1, 3)


def test_groups_must_cover_every_sequence() -> None:
    with pytest.raises(ValueError, match="one group per sequence"):
        _grouped(("coi",))


def test_group_summary_has_one_row_per_group_and_dashes_for_missing_levels() -> None:
    lines = format_group_summary(_grouped(("coi", "ssu", "ssu", "ssu"))).splitlines()

    assert lines[0].split() == [*LEVELS, "support"]
    assert [line.split()[0] for line in lines[2:]] == [OVERALL, "coi", "ssu"]
    assert lines[3].split()[-2:] == ["-", "1"]


def test_group_summary_can_show_another_metric() -> None:
    summary = format_group_summary({OVERALL: _reports()}, metric="macro_f1")

    assert summary.splitlines()[2].split()[LEVELS.index("genus") + 1] == f"{(2 / 3 + 0.5) / 3:.4f}"


def test_an_empty_split_reports_no_metrics() -> None:
    empty = _data().take([])

    grouped = classification_report(
        cast("TaxonomyClassifier", FixedModel()), empty, SPACE, batch_size=8, device=CPU
    )

    assert all(math.isnan(level.accuracy) for level in grouped[OVERALL])
    assert all(level.support == 0 for level in grouped[OVERALL])
