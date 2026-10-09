from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class Marker(StrEnum):
    SSU = "ssu"
    COI = "coi"
    ITS = "its"
    RBCL = "rbcl"
    MATK = "matk"


@dataclass(frozen=True, slots=True)
class LengthRange:
    min_length: int
    max_length: int

    def __post_init__(self) -> None:
        if self.min_length < 1:
            msg = f"min_length must be positive, got {self.min_length}"
            raise ValueError(msg)

        if self.min_length > self.max_length:
            msg = f"min_length ({self.min_length}) exceeds max_length ({self.max_length})"
            raise ValueError(msg)


MARKER_LENGTHS: Final = {
    Marker.COI: LengthRange(min_length=400, max_length=2500),
    Marker.ITS: LengthRange(min_length=200, max_length=1500),
    Marker.RBCL: LengthRange(min_length=400, max_length=1600),
    Marker.MATK: LengthRange(min_length=400, max_length=2500),
}
