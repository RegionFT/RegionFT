from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from RegionFT.core import Bounds


@dataclass(frozen=True)
class ViolationCheck:
    """One checked input and its first witness pair, if it is an IDI."""

    anchor: List[int]
    counterexample: Optional[List[int]]
    anchor_label: Optional[int] = None
    counterexample_label: Optional[int] = None

    @property
    def is_violation(self) -> bool:
        return self.counterexample is not None


class CounterfactualOracle:
    """Detect IDIs by varying only protected attributes."""

    def __init__(self, black_box_model, protected_attrs: Sequence[int]):
        self.black_box_model = black_box_model
        self.protected_attrs = [int(attr) for attr in protected_attrs]
        self.tested_anchors: List[List[int]] = []
        self.violating_pairs: List[List[int]] = []
        # Witness pairs grouped by the phase that discovered their anchor.
        self.violating_pairs_by_phase: Dict[str, List[List[int]]] = {}
        self.tests = 0
        self.violations = 0
        self.phase_tests: Dict[str, int] = {}
        self.phase_violations: Dict[str, int] = {}

    def reset_records(self) -> None:
        self.tested_anchors = []
        self.violating_pairs = []
        self.violating_pairs_by_phase = {}
        self.tests = 0
        self.violations = 0
        self.phase_tests = {}
        self.phase_violations = {}

    def sample_from_bounds(self, bounds: Bounds, rng: random.Random) -> List[int]:
        return [rng.randint(int(lo), int(hi)) for lo, hi in bounds]

    def check(
        self,
        anchor: Sequence[int],
        record: bool = True,
        phase: str = "unknown",
    ) -> ViolationCheck:
        return self.check_many([anchor], record=record, phase=phase)[0]

    def check_many(
        self,
        anchors: Sequence[Sequence[int]],
        record: bool = True,
        phase: str = "unknown",
    ) -> List[ViolationCheck]:
        """Batch-check inputs, counting each as one test and keeping its first witness pair."""

        anchors = [[int(value) for value in anchor] for anchor in anchors]
        if not anchors:
            return []

        candidate_groups = [self._counterfactual_candidates(anchor) for anchor in anchors]
        batch_rows: List[List[int]] = []
        offsets: List[Tuple[int, int]] = []
        for anchor, candidates in zip(anchors, candidate_groups):
            start = len(batch_rows)
            batch_rows.append(anchor)
            batch_rows.extend(candidates)
            offsets.append((start, 1 + len(candidates)))

        labels = list(self.black_box_model.predict(batch_rows))
        if len(labels) != len(batch_rows):
            raise ValueError("black_box_model.predict returned an unexpected number of labels")
        results: List[ViolationCheck] = []
        for anchor, candidates, (start, length) in zip(anchors, candidate_groups, offsets):
            group_labels = labels[start : start + length]
            anchor_label = int(group_labels[0])
            counterexample = None
            counterexample_label = None
            for candidate, candidate_label in zip(candidates, group_labels[1:]):
                candidate_label = int(candidate_label)
                if candidate_label != anchor_label:
                    counterexample = candidate
                    counterexample_label = candidate_label
                    break
            result = ViolationCheck(
                anchor=anchor,
                counterexample=counterexample,
                anchor_label=anchor_label,
                counterexample_label=counterexample_label,
            )
            if record:
                self._record(result, phase)
            results.append(result)
        return results

    def _counterfactual_candidates(self, anchor: Sequence[int]) -> List[List[int]]:
        protected_value_ranges = []
        for attr_idx in self.protected_attrs:
            lo, hi = self.black_box_model.data_range[attr_idx]
            protected_value_ranges.append(range(int(lo), int(hi) + 1))

        if all(len(values) <= 1 for values in protected_value_ranges):
            return []

        anchor_values = tuple(int(anchor[attr_idx]) for attr_idx in self.protected_attrs)
        candidates = []
        for protected_values in product(*protected_value_ranges):
            protected_values = tuple(int(value) for value in protected_values)
            if protected_values == anchor_values:
                continue
            candidate = list(anchor)
            for attr_idx, value in zip(self.protected_attrs, protected_values):
                candidate[attr_idx] = value
            candidates.append(candidate)
        return candidates

    def _record(self, result: ViolationCheck, phase: str) -> None:
        self.tested_anchors.append(result.anchor)
        self.tests += 1
        self.phase_tests[phase] = self.phase_tests.get(phase, 0) + 1
        if result.counterexample is not None:
            self.violations += 1
            self.phase_violations[phase] = self.phase_violations.get(phase, 0) + 1
            # Store each witness pair as two rows with predicted labels appended.
            anchor_row = result.anchor + [int(result.anchor_label)]
            counter_row = result.counterexample + [int(result.counterexample_label)]
            self.violating_pairs.append(anchor_row)
            self.violating_pairs.append(counter_row)
            bucket = self.violating_pairs_by_phase.setdefault(phase, [])
            bucket.append(anchor_row)
            bucket.append(counter_row)
