"""A finite typed DSL: no evaluated Python, unrestricted SQL or target-row fields."""

from dataclasses import asdict, dataclass
import logging
from typing import Sequence

import polars as pl

from src.models.baseline.feature_engineering import ENTITY_KEYS, EVENT_TYPES, FeatureSpec, manual_specs
from src.utils.reproducibility import stable_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RejectedFeature:
    """Auditable reason why a proposed historical primitive is unavailable."""

    specification: dict
    reason: str


class AutoFeatureGenerator:
    """Generate bounded window/entity/event/aggregation candidates and validate each."""

    def __init__(self, dataset: str, *, windows_days: Sequence[int] = (1, 3, 7, 14, 30),
                 max_candidate_features: int = 500, max_feature_depth: int = 2, seed: int = 42) -> None:
        if dataset not in EVENT_TYPES or not windows_days or max_candidate_features < 1 or max_candidate_features > 500 or max_feature_depth not in (1, 2):
            raise ValueError("invalid AutoML dataset, feature count, windows or DSL depth")
        for window in windows_days:
            FeatureSpec("user", window, "count").validate(dataset)
        self.dataset = dataset
        self.windows_days = tuple(sorted(set(windows_days)))
        self.max_candidate_features = max_candidate_features
        self.max_feature_depth = max_feature_depth
        self.seed = seed
        self.created = 0
        self.rejected: list[RejectedFeature] = []

    def validate(self, candidates: Sequence[FeatureSpec]) -> list[FeatureSpec]:
        """Reject unchecked proposals while retaining reasons for every rejection."""
        accepted = []
        self.rejected = []
        names: set[str] = set()
        for candidate in candidates:
            try:
                if type(candidate) is not FeatureSpec:
                    raise ValueError("only typed FeatureSpec objects are allowed")
                candidate.validate(self.dataset)
                if self.max_feature_depth == 1 and candidate.aggregation in {"positive_rate", "target_encoding"}:
                    raise ValueError("a ratio/smoothed encoding requires depth two")
                if candidate.name in names:
                    raise ValueError("duplicate feature name")
                names.add(candidate.name)
                accepted.append(candidate)
            except ValueError as exc:
                self.rejected.append(RejectedFeature(asdict(candidate) if isinstance(candidate, FeatureSpec) else {"type": type(candidate).__name__}, str(exc)))
        return accepted

    def generate(self) -> list[FeatureSpec]:
        """Return up to 500 reproducible candidates, preserving the manual reference set."""
        manual = manual_specs(self.dataset)
        grid = [FeatureSpec(entity, day, aggregation, kind) for entity in ENTITY_KEYS
                for day in self.windows_days for aggregation in ("count", "sum", "mean", "min", "max")
                for kind in (None, *EVENT_TYPES[self.dataset])]
        unique = {spec.name: spec for spec in [*manual, *grid]}
        self.created = len(unique)
        valid = self.validate(list(unique.values()))
        baseline_names = {spec.name for spec in manual}
        retained_manual = sorted((s for s in valid if s.name in baseline_names), key=lambda s: s.name)
        extras = sorted((s for s in valid if s.name not in baseline_names), key=lambda s: (stable_seed(self.seed, self.dataset, s.name), s.name))
        selected = (retained_manual + extras)[:self.max_candidate_features]
        LOGGER.info("AutoML dataset=%s proposed=%s rejected=%s candidates=%s", self.dataset, self.created, len(self.rejected), len(selected))
        return selected

    def audit(self, candidates: Sequence[FeatureSpec]) -> pl.DataFrame:
        """Describe the complete bounded search, including each retained definition."""
        return pl.DataFrame([{"feature": s.name, "entity": s.entity, "window_days": s.window_days,
            "aggregation": s.aggregation, "event_type": s.event_type, "proposed_total": self.created,
            "rejected_total": len(self.rejected), "candidate_total": len(candidates), "max_depth": self.max_feature_depth}
            for s in candidates], schema={"feature": pl.String, "entity": pl.String, "window_days": pl.Int64,
                "aggregation": pl.String, "event_type": pl.String, "proposed_total": pl.Int64,
                "rejected_total": pl.Int64, "candidate_total": pl.Int64, "max_depth": pl.Int64})
