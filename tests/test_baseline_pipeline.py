"""Complete shared-feature, HPO, calibration and test prediction integration."""

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from src.generator.artifacts import ArtifactBundle
from src.models.baseline.feature_engineering import FeatureSpec
from src.models.baseline.pipeline import collect_budgeted, run_gbdt
from src.generator.storage import write_batches
from src.utils.reproducibility import config_hash, file_hash
from tests.helpers import model_bundle


def test_baseline_experiment_and_bundle_integrity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, root = model_bundle(tmp_path)
    bundle = ArtifactBundle(root)
    assert bundle.events().select(pl.len()).collect().item() == 200
    assert bundle.scan("metadata/items.parquet").collect().is_empty()
    with pytest.raises(ValueError, match="absent"):
        bundle.path("../raw.csv")
    assert bundle.graph().select(pl.len()).collect().item() == 200
    original_entries = bundle.entries
    bundle.entries = {k: v for k, v in original_entries.items() if not k.startswith("model_specific/graph/")}
    with pytest.raises(ValueError, match="graph"):
        bundle.graph()
    bundle.entries = original_entries
    with pytest.raises(ValueError, match="config"):
        ArtifactBundle(root, "wrong")
    with pytest.raises(ValueError, match="smoke"):
        run_gbdt(config, "retailrocket", processed_root=root)
    specs = [FeatureSpec("user", 1, "count"), FeatureSpec("item", None, "target_encoding"), FeatureSpec("user_item", None, "count")]
    result = run_gbdt(config, "retailrocket", processed_root=root, smoke=True, specs=specs)
    scores = pl.read_parquet(result / "seed_42/test_predictions.parquet")
    assert scores.height == 500 and scores["score"].is_finite().all()
    assert scores["probability"].is_between(0, 1).all()
    metrics = pl.read_parquet(result / "seed_42/test_metrics.parquet")
    assert metrics["mode"].item() == "smoke" and metrics["n_groups"].item() == 5
    assert pl.read_parquet(result / "hpo_trials.parquet").height == 2
    import src.models.baseline.pipeline as pipeline
    monkeypatch.setattr(pipeline.psutil, "virtual_memory", lambda: SimpleNamespace(available=1))
    with pytest.raises(MemoryError, match="No training rows were dropped"):
        collect_budgeted(pl.DataFrame({"x": [1]}).lazy(), 100)
    path = root / "metadata/items.parquet"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ArtifactBundle(root).path("metadata/items.parquet")


def test_full_hpo_does_not_silently_shorten_search_after_budgeted_fits(tmp_path: Path, monkeypatch):
    import datetime as clock
    import optuna.study._optimize as optimizer
    class ScoringClock:
        calls = 0
        @classmethod
        def now(cls, *args, **kwargs):
            cls.calls += 1
            return clock.datetime.now(*args, **kwargs) + clock.timedelta(hours=3 * (cls.calls - 1))
    # Simulate validation/scoring overhead consuming a global timeout while each
    # actual native fit remains within its valid integer-hour training budget.
    monkeypatch.setattr(optimizer, "datetime", SimpleNamespace(datetime=ScoringClock))
    config, root = model_bundle(tmp_path)
    training = config.runtime.training.model_copy(update={"hpo_trials": 3})
    config = config.model_copy(update={"runtime": config.runtime.model_copy(update={"training": training})})
    digest = config_hash(config)
    state_path = root / "audit/generation_state.parquet"
    state = pl.read_parquet(state_path).with_columns(pl.lit("full").alias("mode"), pl.lit(digest).alias("config_hash"))
    write_batches(state_path, [state], dict(state.schema))
    manifest_path = root / "audit/artifact_manifest.parquet"
    manifest = pl.read_parquet(manifest_path).with_columns(pl.lit(digest).alias("config_hash"),
        pl.when(pl.col("artifact_path") == "audit/generation_state.parquet").then(pl.lit(file_hash(state_path))).otherwise(pl.col("sha256")).alias("sha256"))
    write_batches(manifest_path, [manifest], dict(manifest.schema))
    specs = [FeatureSpec("user", 1, "count"), FeatureSpec("item", None, "target_encoding"), FeatureSpec("user_item", None, "count")]
    result = run_gbdt(config, "retailrocket", processed_root=root, specs=specs)
    assert pl.read_parquet(result / "hpo_trials.parquet").height == 3
    provenance = pl.read_parquet(result / "run_manifest.parquet")
    assert provenance["hpo_attempted_trials"].item() == 3 and provenance["device"].item() == "cpu"
    assert provenance["environment"].item() and provenance["training_seeds"].item().to_list() == [42, 43, 44]
    assert all((result / f"seed_{seed}/test_predictions.parquet").exists() for seed in (42, 43, 44))
