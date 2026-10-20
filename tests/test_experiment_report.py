"""Collector verification uses synthetic prediction fixtures, never published model results."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import polars as pl
import pytest
from src.evaluation.experiment_report import (MODELS, bootstrap_contrast, cohort_registry,
    collect_dataset, holm_adjust, render_report, save_collection, _prefix_counts)
from src.evaluation.metrics import PREDICTION_SCHEMA, compare_models
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.utils.reproducibility import config_hash, file_hash
from tests.helpers import model_bundle


def fixture_runs(config, bundle):
    """Write deliberate known-rank toy predictions inside the temporary test directory."""
    truth = bundle.scan("evaluation/test/eval_candidates.parquet").collect()
    for model in MODELS:
        root = config.project.experiments_dir / model / "retailrocket/smoke"
        predictions = truth.select("eval_group_id", "candidate_item_id").with_columns(
            truth["candidate_label"].cast(pl.Float64).alias("score"),
            (.05 + truth["candidate_label"] * .9).alias("probability"),
            pl.lit(model).alias("model"), pl.lit(42, dtype=pl.Int64).alias("training_seed")).cast(PREDICTION_SCHEMA)
        write_batches(root / "seed_42/test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
        metrics = compare_models(truth, {model: predictions}).with_columns(pl.lit(config_hash(config)).alias("config_hash"), pl.lit(42).alias("training_seed"))
        write_batches(root / "seed_42/test_metrics.parquet", [metrics], dict(metrics.schema))
        manifest = pl.DataFrame({"model": [model], "dataset": ["retailrocket"], "mode": ["smoke"],
            "config_hash": [config_hash(config)], "parent_manifest_hash": [file_hash(bundle.manifest_path)], "wall_seconds": [1.]})
        write_batches(root / "run_manifest.parquet", [manifest], dict(manifest.schema))


def test_verified_collection_cohorts_and_report(tmp_path: Path):
    config, processed = model_bundle(tmp_path)
    bundle = ArtifactBundle(processed)
    fixture_runs(config, bundle)
    result = collect_dataset(config, "retailrocket", mode="smoke", processed_root=processed, repetitions=30)
    cells, metrics, comparisons, coverage = result
    assert cells.height == 5 and (cells["status"] == "complete").all()
    assert (cells["NDCG@10_mean"] == 1.).all()
    assert cells["NDCG@10_std"].null_count() == 5
    assert cells["peak_rss_bytes"].null_count() == 5 and (cells["artifact_bytes"] > 0).all()
    assert cells["human_minutes"].null_count() == 5 and cells["latency_p95_ms"].null_count() == 5
    assert comparisons.height == 4 and (comparisons["holm_p"] == 1.).all()
    assert metrics.filter(pl.col("cohort") == "train_cold_user")["value"].null_count() == 75
    assert coverage.filter(pl.col("cohort") == "all")["n_groups"].item() == 5
    cohorts = cohort_registry(bundle, "retailrocket")
    future = bundle.events().filter(pl.col("split") == "test").with_columns(pl.lit(1000, dtype=pl.Int64).alias("timestamp"), (pl.col("user_id") + 10000).alias("user_id"))
    mutated = SimpleNamespace(events=lambda: pl.concat([bundle.events(), future]), scan=bundle.scan)
    assert cohorts.equals(cohort_registry(mutated, "retailrocket"))
    absent = collect_dataset(config, "retailrocket", processed_root=tmp_path / "absent")
    assert (absent[0]["status"] == "missing_dataset").all()
    wrong_mode = collect_dataset(config, "retailrocket", mode="full", processed_root=processed)
    assert (wrong_mode[0]["status"] == "invalid_dataset").all()
    with pytest.raises(ValueError):
        collect_dataset(config, "retailrocket", mode="wrong")
    save_collection(tmp_path / "smoke", [result])
    save_collection(tmp_path / "full", [absent])
    render_report(tmp_path / "full", tmp_path / "smoke", tmp_path / "results.md")
    assert "не выполнена" in (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Ограниченный smoke" in (tmp_path / "results.md").read_text(encoding="utf-8")
    # A stale run with old scores must not survive an interrupted rerun.
    state = pl.DataFrame({"status": ["running"]})
    root = config.project.experiments_dir / "adv2/retailrocket/smoke"
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    invalid = collect_dataset(config, "retailrocket", mode="smoke", processed_root=processed, repetitions=10)
    assert invalid[0].filter(pl.col("model") == "adv2")["status"].item() == "invalid_run"
    predictions = pl.read_parquet(root / "seed_42/test_predictions.parquet").head(1)
    write_batches(root / "seed_42/test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
    state = state.with_columns(pl.lit("complete").alias("status"))
    write_batches(root / "run_state.parquet", [state], dict(state.schema))
    invalid = collect_dataset(config, "retailrocket", mode="smoke", processed_root=processed, repetitions=10)
    assert invalid[0].filter(pl.col("model") == "adv2")["status"].item() == "invalid_run"


def test_holm_and_cluster_pairing():
    assert np.allclose(holm_adjust([.01, .04, .03, .2]), [.04, .09, .09, .2])
    for value in ([float("nan")], [-.1], [1.1]):
        with pytest.raises(ValueError):
            holm_adjust(value)
    groups = pl.DataFrame({"user_id": [1, 1, 2], "difference": [.2, .1, .3]})
    result = bootstrap_contrast(groups, repetitions=100, seed=42)
    assert result["delta"] == pytest.approx(.2) and result["ci_lower"] > 0 and result["p_value"] < .05
    assert result == bootstrap_contrast(groups.reverse(), repetitions=100, seed=42)
    with pytest.raises(ValueError):
        bootstrap_contrast(groups, repetitions=1)


def test_full_requires_every_seed_and_tracks_measured_resources(tmp_path: Path):
    from src.evaluation.execution import JOURNAL_SCHEMA
    config, processed = model_bundle(tmp_path)
    state_path = processed / "audit/generation_state.parquet"
    state = pl.read_parquet(state_path).with_columns(pl.lit("full").alias("mode"))
    write_batches(state_path, [state], dict(state.schema))
    manifest_path = processed / "audit/artifact_manifest.parquet"
    manifest = pl.read_parquet(manifest_path).with_columns(pl.when(pl.col("artifact_path") == "audit/generation_state.parquet")
        .then(pl.lit(file_hash(state_path))).otherwise(pl.col("sha256")).alias("sha256"))
    write_batches(manifest_path, [manifest], dict(manifest.schema))
    bundle = ArtifactBundle(processed)
    fixture_runs(config, bundle)
    for model in MODELS:
        smoke = config.project.experiments_dir / model / "retailrocket/smoke"
        full = smoke.parent / "full"
        provenance = pl.read_parquet(smoke / "run_manifest.parquet").with_columns(pl.lit("full").alias("mode"),
            pl.lit(1024).alias("cuda_peak_allocated_bytes"), pl.lit(2048).alias("cuda_peak_reserved_bytes"))
        write_batches(full / "run_manifest.parquet", [provenance], dict(provenance.schema))
        for seed in config.runtime.training_seeds:
            predictions = pl.read_parquet(smoke / "seed_42/test_predictions.parquet").with_columns(pl.lit(seed).alias("training_seed")).cast(PREDICTION_SCHEMA)
            metrics = pl.read_parquet(smoke / "seed_42/test_metrics.parquet").with_columns(pl.lit(seed).alias("training_seed"))
            write_batches(full / f"seed_{seed}/test_predictions.parquet", [predictions], PREDICTION_SCHEMA)
            write_batches(full / f"seed_{seed}/test_metrics.parquet", [metrics], dict(metrics.schema))
    row = dict.fromkeys(JOURNAL_SCHEMA)
    row.update(dataset="retailrocket", stage="model", model="baseline", attempt=1, status="complete",
        config_hash=config_hash(config), peak_rss_bytes=4096, cpu_seconds=2., measurement="toy fixture measurement")
    journal = config.project.experiments_dir / "execution/journal.parquet"
    write_batches(journal, [pl.DataFrame([row], schema=JOURNAL_SCHEMA)], JOURNAL_SCHEMA)
    result = collect_dataset(config, "retailrocket", processed_root=processed, repetitions=10)
    cells = result[0]
    assert (cells["status"] == "complete").all() and (cells["n_seeds"] == 3).all()
    assert (cells["NDCG@10_std"] == 0).all()
    assert cells.filter(pl.col("model") == "baseline")["peak_rss_bytes"].item() == 4096
    assert (cells["cuda_peak_allocated_bytes"] == 1024).all()
    save_collection(tmp_path / "full", [result])
    render_report(tmp_path / "full", None, tmp_path / "results.md")
    assert "95% CI" in (tmp_path / "results.md").read_text(encoding="utf-8")
    (config.project.experiments_dir / "adv4/retailrocket/full/seed_44/test_predictions.parquet").unlink()
    row["status"] = "running"
    write_batches(journal, [pl.DataFrame([row], schema=JOURNAL_SCHEMA)], JOURNAL_SCHEMA)
    invalid = collect_dataset(config, "retailrocket", processed_root=processed, repetitions=10)[0]
    assert invalid.filter(pl.col("model") == "baseline")["status"].item() == "running_run"
    assert invalid.filter(pl.col("model") == "adv4")["status"].item() != "complete"


def test_streamed_prefix_counts_match_independent_formula_across_ties():
    rng = np.random.default_rng(42)
    events = pl.DataFrame({"user_id": rng.integers(1, 8, 500), "timestamp": rng.integers(1, 20, 500)})
    queries = pl.DataFrame({"user_id": [999, 1, 2, 2, 1, 3, 7, 7, 8], "timestamp": [100, 0, 10, 10, 100, 5, 19, 20, 1]})
    expected = np.array([events.filter((pl.col("user_id") == identity) & (pl.col("timestamp") < timestamp)).height for identity, timestamp in queries.iter_rows()])
    for chunk in (3, 1000):
        assert np.array_equal(_prefix_counts(events.lazy(), queries, "user_id", chunk_rows=chunk), expected)
    assert np.array_equal(_prefix_counts(events.head(0).lazy(), queries, "user_id"), np.zeros(queries.height))
    assert _prefix_counts(events.lazy(), queries.head(0), "user_id").size == 0
