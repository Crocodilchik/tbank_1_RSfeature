"""Fail-closed structural and temporal checks for generated artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import polars as pl

from src.generator.schemas import EVENTS_SCHEMA, EVENTS_NULLABLE, SEQUENCES_SCHEMA
from src.generator.splitting.temporal_splitter import primary_expression

GATES_SCHEMA = {"gate": pl.String, "passed": pl.Boolean, "checked_rows": pl.Int64, "violations": pl.Int64, "details": pl.String}


class QualityGateError(ValueError):
    """An artifact cannot be published as a strict PIT result."""


@dataclass(frozen=True)
class GateResult:
    """Measured result, never a substituted model metric."""

    gate: str
    passed: bool
    checked_rows: int
    violations: int
    details: str


def schema_gate(frame: pl.LazyFrame, schema: dict, nullable: set, keys: list[str], name: str) -> GateResult:
    """Check exact physical schema, nonnull fields and key uniqueness on all rows."""
    actual = frame.collect_schema()
    if dict(actual) != schema:
        return GateResult(name, False, 0, 1, f"schema mismatch: {actual}")
    required = [c for c in schema if c not in nullable]
    counts = frame.select(pl.len().alias("n"), pl.any_horizontal(pl.col(c).is_null() for c in required).sum().alias("null_rows"),
                          pl.struct(keys).n_unique().alias("unique_keys")).collect(engine="streaming").row(0)
    n, nulls, unique = counts
    violations = int(nulls) + n - unique
    return GateResult(name, violations == 0, n, violations, "exact schema; required fields; unique key")


def temporal_gate(events: pl.LazyFrame) -> GateResult:
    """Check split ordering and cutoffs on the complete real event stream."""
    ranges = events.group_by("split").agg(pl.col("timestamp").min().alias("minimum"), pl.col("timestamp").max().alias("maximum"), pl.len().alias("n")).collect()
    mapping = {r["split"]: r for r in ranges.iter_rows(named=True)}
    violations = 0
    if set(mapping) != {"train", "valid", "test"}:
        violations += 1
    else:
        violations += int(mapping["train"]["maximum"] >= mapping["valid"]["minimum"])
        violations += int(mapping["valid"]["maximum"] >= mapping["test"]["minimum"])
    invalid = events.filter((pl.col("timestamp") != pl.col("target_cutoff_ts")) | pl.col("is_synthetic") |
                            ~pl.col("label").is_in([0, 1, 2])).select(pl.len()).collect().item()
    violations += invalid
    return GateResult("temporal_splits", violations == 0, int(ranges["n"].sum()), violations, "half-open splits; real events; target cutoff equals timestamp")


def sequence_gate(samples: pl.LazyFrame, events: pl.LazyFrame, max_history: int = 50, seed: int = 42) -> list[GateResult]:
    """Check every sequence structurally and compare sampled histories with independent prefixes."""
    structural = 0
    count = 0
    reservoir: list[dict] = []
    rng = np.random.default_rng(seed)
    for batch in samples.collect_batches(chunk_size=100_000):
        for row in batch.iter_rows(named=True):
            count += 1
            histories = [row[c] for c in ("history_event_ids", "history_item_ids", "history_timestamps", "history_labels", "history_category_ids", "history_time_deltas")]
            length = len(histories[0])
            valid = all(len(h) == length for h in histories) and length <= max_history
            valid &= row["cutoff_ts"] == row["target_timestamp"] and row["target_event_id"] not in histories[0]
            valid &= all(t < row["target_timestamp"] for t in histories[2])
            valid &= histories[5] == [row["target_timestamp"] - t for t in histories[2]]
            valid &= row["max_history_timestamp"] == (max(histories[2]) if length else None)
            structural += not valid
            if len(reservoir) < 100:
                reservoir.append(row)
            else:
                index = int(rng.integers(count))
                if index < 100:
                    reservoir[index] = row
    reference_bad = 0
    if reservoir:
        queries = pl.DataFrame({"user_id": [r["user_id"] for r in reservoir], "target_event_id": [r["target_event_id"] for r in reservoir], "T": [r["target_timestamp"] for r in reservoir]})
        wanted = queries.select("user_id").unique().lazy()
        prefixes = events.join(wanted, on="user_id", how="semi").select("user_id", "timestamp", "event_id", "item_id", "label").join(queries.lazy(), on="user_id").filter(pl.col("timestamp") < pl.col("T"))
        reference = prefixes.sort(["timestamp", "event_id"]).group_by("target_event_id").agg(
            pl.col("event_id").tail(max_history).alias("ids"), pl.col("item_id").tail(max_history).alias("items"),
            pl.col("timestamp").tail(max_history).alias("times"), pl.col("label").tail(max_history).alias("labels")).collect(engine="streaming")
        lookup = {r["target_event_id"]: r for r in reference.iter_rows(named=True)}
        for row in reservoir:
            expected = lookup.get(row["target_event_id"], {"ids": [], "items": [], "times": [], "labels": []})
            reference_bad += any(row[field] != expected[key] for field, key in (("history_event_ids", "ids"), ("history_item_ids", "items"), ("history_timestamps", "times"), ("history_labels", "labels")))
    return [GateResult("sequence_structural", structural == 0, count, int(structural), "all samples; strict seconds; empty history and deltas"),
            GateResult("sequence_prefix_reference", reference_bad == 0, len(reservoir), reference_bad, "independent real-event prefix, seed42, up to100 per split")]


def candidate_gate(candidates: pl.LazyFrame) -> GateResult:
    """Validate full groups, answer identity and prefix-catalog timestamps."""
    groups = candidates.group_by("eval_group_id").agg(pl.len().alias("n"), pl.col("candidate_item_id").n_unique().alias("unique"),
        pl.col("candidate_label").sum().alias("positives"),
        (~pl.col("candidate_label").is_in([0, 1])).sum().alias("bad_label"),
        ((pl.col("candidate_label") == 1) != (pl.col("candidate_item_id") == pl.col("ground_truth_item_id"))).sum().alias("wrong_answer"),
        (pl.col("catalog_snapshot_ts") >= pl.col("target_timestamp")).sum().alias("future"))
    row = groups.select(pl.len().alias("groups"), ((pl.col("n") != 100) | (pl.col("unique") != 100) | (pl.col("positives") != 1) | (pl.col("bad_label") > 0) | (pl.col("wrong_answer") > 0) | (pl.col("future") > 0)).sum().alias("bad")).collect().row(0)
    return GateResult("evaluation_candidates", row[1] == 0, row[0], row[1], "100 unique; exactly one identified GT; known prefix catalog")


def candidate_exclusion_gate(candidates: pl.LazyFrame, events: pl.LazyFrame, dataset: str,
                             split: str, *, fallback: bool = False, exclude_future: bool = False) -> GateResult:
    """Independently check all negative pairs against the mandatory evaluator exclusions."""
    negatives = candidates.filter(pl.col("candidate_label") == 0).select("eval_group_id", "user_id", pl.col("candidate_item_id").alias("item_id"), "target_timestamp")
    n = negatives.select(pl.len()).collect().item()
    if n == 0:
        return GateResult("evaluation_exclusions", True, 0, 0, "empty separate cohort; no negative pairs")
    pairs = negatives.select("user_id", "item_id").unique()
    relevant = events.join(pairs, on=["user_id", "item_id"], how="semi").filter(~pl.col("is_synthetic"))
    excluded = relevant.filter((pl.col("split") == split) & primary_expression(dataset, fallback)).select("user_id", "item_id").unique()
    if dataset == "movielens":
        excluded = relevant.select("user_id", "item_id").unique()
    bad = negatives.join(excluded, on=["user_id", "item_id"], how="semi").select(pl.len()).collect().item()
    if dataset == "yoochoose" or exclude_future:
        range_facts = relevant.filter(primary_expression(dataset, fallback)) if exclude_future and dataset != "yoochoose" else relevant
        limits = range_facts.group_by(["user_id", "item_id"]).agg(pl.col("timestamp").min().alias("minimum"), pl.col("timestamp").max().alias("maximum"))
        predicate = pl.col("minimum") < pl.col("target_timestamp") if dataset == "yoochoose" else pl.lit(False)
        if exclude_future:
            predicate |= pl.col("maximum") > pl.col("target_timestamp")
        bad += negatives.join(limits, on=["user_id", "item_id"]).filter(predicate).select(pl.len()).collect().item()
    return GateResult("evaluation_exclusions", bad == 0, n, bad, "all negatives; split positives; dataset repeat policy; optional future oracle")


def negative_gate(negatives: pl.LazyFrame, events: pl.LazyFrame) -> GateResult:
    """Check every frozen negative against context, catalog and real past interactions."""
    invalid = negatives.filter((pl.col("label") != 0) | (pl.col("timestamp") != pl.col("context_timestamp")) |
        (pl.col("catalog_snapshot_ts") >= pl.col("context_timestamp")) |
        (pl.col("future_positive_conflict") != pl.col("conflict_first_future_ts").is_not_null()) |
        (pl.col("conflict_first_future_ts").is_not_null() & (pl.col("conflict_first_future_ts") <= pl.col("context_timestamp"))))
    bad = invalid.select(pl.len()).collect().item()
    contexts = events.select(pl.col("event_id").alias("context_event_id"), pl.col("user_id").alias("expected_user"),
                             pl.col("timestamp").alias("expected_time"), pl.col("item_id").alias("positive_item"), "split")
    wrong_context = negatives.join(contexts, on="context_event_id", how="left").filter(
        pl.col("expected_user").is_null() | (pl.col("user_id") != pl.col("expected_user")) |
        (pl.col("context_timestamp") != pl.col("expected_time")) | (pl.col("item_id") == pl.col("positive_item")) | (pl.col("split") != "train"))
    bad += wrong_context.select(pl.len()).collect().item()
    pairs = negatives.select("user_id", "item_id").unique()
    matching = events.join(pairs, on=["user_id", "item_id"], how="semi").select("user_id", "item_id", pl.col("timestamp").alias("real_ts"))
    first_pair = matching.group_by(["user_id", "item_id"]).agg(pl.col("real_ts").min().alias("first_pair_ts"))
    bad += negatives.join(first_pair, on=["user_id", "item_id"], how="left").filter(pl.col("first_pair_ts") <= pl.col("context_timestamp")).select(pl.len()).collect().item()
    future = negatives.sort("context_timestamp").join_asof(matching.sort("real_ts"), left_on="context_timestamp", right_on="real_ts",
        by=["user_id", "item_id"], strategy="forward", allow_exact_matches=False, check_sortedness=False)
    bad += future.filter(~pl.col("real_ts").eq_missing(pl.col("conflict_first_future_ts"))).select(pl.len()).collect().item()
    first_seen = events.group_by("item_id").agg(pl.col("timestamp").min().alias("first_seen"))
    bad += negatives.join(first_seen, on="item_id", how="left").filter(pl.col("first_seen").is_null() | (pl.col("first_seen") >= pl.col("context_timestamp"))).select(pl.len()).collect().item()
    count = negatives.select(pl.len()).collect().item()
    return GateResult("training_negatives", bad == 0, count, bad, "all contexts train; real history exclusions; prefix catalog; diagnostic conflict semantics")


def metadata_gate(metadata: pl.LazyFrame) -> GateResult:
    """Validate availability markers without trusting future valid_to fields."""
    static = pl.col("availability_class") == "static_assumed"
    dynamic = pl.col("availability_class").is_in(["observed_snapshot", "derived_pit"])
    invalid = metadata.filter(~(static | dynamic) | (pl.col("field_value") == "null") |
        (static & ((pl.col("valid_from") != -(2**63)) | pl.col("snapshot_timestamp").is_not_null())) |
        (dynamic & (pl.col("snapshot_timestamp").is_null() | (pl.col("snapshot_timestamp") != pl.col("valid_from")))))
    n = metadata.select(pl.len()).collect().item()
    bad = invalid.select(pl.len()).collect().item()
    return GateResult("metadata_availability", bad == 0, n, bad, "static sentinel; observed/derived snapshot provenance; no logical null value")


def graph_gate(edges: pl.LazyFrame) -> GateResult:
    """Validate complete delta logs; per-query graphs are selected separately."""
    bad = edges.filter((pl.col("valid_from") != pl.col("timestamp")) | ~pl.col("src_type").is_in(["User", "Item", "Category", "Brand"]) |
        ~pl.col("dst_type").is_in(["User", "Item", "Category", "Brand"]) | ~pl.col("source").is_in(["real", "synthetic"]) |
        (pl.col("weight").is_not_null() & ~pl.col("weight").is_finite())).select(pl.len()).collect().item()
    count = edges.select(pl.len()).collect().item()
    return GateResult("graph_delta_provenance", bad == 0, count, bad, "delta timestamps and types; graph_at applies strict per-target cutoff and relation as-of")


def prefix_invariance(feature_builder: Callable[[pl.DataFrame, dict], dict], events: pl.DataFrame, target: dict) -> GateResult:
    """Compare a feature builder against a world with all future events removed."""
    full = feature_builder(events, target)
    prefix = feature_builder(events.filter(pl.col("timestamp") < target["timestamp"]), target)
    bad = set(full) != set(prefix)
    if not bad:
        for key in full:
            a, b = full[key], prefix[key]
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                bad |= not bool(np.isclose(a, b, rtol=1e-6, atol=1e-8, equal_nan=False))
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                bad |= a.shape != b.shape or not bool(np.allclose(a, b, rtol=1e-6, atol=1e-8))
            else:
                bad |= a != b
    return GateResult("prefix_invariance", not bad, 1, int(bad), "same query and static inputs; all events>=T removed")


def assert_gates(results: list[GateResult]) -> pl.DataFrame:
    """Return the auditable gate table or raise with every failed condition."""
    failures = [r for r in results if not r.passed or r.violations]
    if failures:
        raise QualityGateError("; ".join(f"{r.gate}: {r.violations} violations ({r.details})" for r in failures))
    return pl.DataFrame([asdict(r) for r in results], schema=GATES_SCHEMA)
