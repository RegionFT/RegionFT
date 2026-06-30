from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


Bounds = List[Tuple[int, int]]


@dataclass(frozen=True)
class RegionStats:
    """Estimated IDI rate for one final region."""

    target_samples: int
    samples: int
    violations: int

    @property
    def ratio(self) -> float:
        return self.violations / float(self.samples) if self.samples else 0.0

    @property
    def gini(self) -> float:
        ratio = self.ratio
        return 2.0 * ratio * (1.0 - ratio)


@dataclass
class Region:
    """A rectangular region in the final partition."""

    region_id: int
    bounds: Bounds
    size: int
    stats: RegionStats
    depth: int = 0
    parent_id: int = -1
    split_history: List[str] = field(default_factory=list)
    best_attr: Optional[int] = None
    best_cut: Optional[int] = None
    best_gain: float = 0.0


def normalize_bounds(bounds: Sequence[Sequence[int]]) -> Bounds:
    return [(int(lo), int(hi)) for lo, hi in bounds]


def region_size(bounds: Sequence[Tuple[int, int]]) -> int:
    size = 1
    for lo, hi in bounds:
        size *= max(0, int(hi) - int(lo) + 1)
    return size


def bounds_to_text(bounds: Sequence[Tuple[int, int]]) -> str:
    return ";".join(f"{int(lo)}:{int(hi)}" for lo, hi in bounds)
