"""Actual teacher/student fitting, OOF lineage and student-only inference."""
from pathlib import Path
import numpy as np
import polars as pl
import pytest
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE
from src.models.baseline.booster import BoosterModel
from src.models.baseline.feature_engineering import FeatureSpec
from src.models.common import temporal_graph
from src.models.distillation.teacher_student import TeacherArtifact, choose_alpha, mixed_targets, run_distillation, validate_teacher_rows
from tests.helpers import model_bundle

@pytest.mark.parametrize("mode", ["score_feature_stacking", "soft_target_distillation"])
def test_teacher_student_end_to_end_and_rejections(tmp_path: Path, monkeypatch, mode):
    config, processed = model_bundle(tmp_path)
    def small_graph(**kw):
        kw.update(hidden_size=8, hash_slots=128)
        return HeterogeneousGraphSAGE(**kw)
    monkeypatch.setattr(temporal_graph, "HeterogeneousGraphSAGE", small_graph)
    specs = [FeatureSpec("user", 1, "count"), FeatureSpec("item", None, "target_encoding"), FeatureSpec("user_item", None, "count")]
    root = run_distillation(config, "retailrocket", mode=mode, processed_root=processed, smoke=True, specs=specs)
    assert pl.read_parquet(root / "run_state.parquet")["status"].item() == "complete"
    assert pl.read_parquet(root / "seed_42/test_metrics.parquet")["n_groups"].item() == 5
    model = BoosterModel.load(root / "seed_42")
    features_path = next(path for path in (processed / "model_specific/flat").glob("features_*.parquet") if not path.stem.endswith(("_gates", "_manifest")))
    inputs = pl.read_parquet(features_path).filter(pl.col("split") == "test")
    if mode == "soft_target_distillation":
        assert all(not feature.startswith("teacher_") for feature in model.features)
        assert np.isfinite(model.predict(inputs)).all()
        alpha = pl.read_parquet(root / "alpha_selection.parquet")
        assert (alpha["fit_end_ts"] < alpha["holdout_start_ts"]).all()
    else:
        assert {"teacher_score", "teacher_available"} <= set(model.features)
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            model.predict(inputs)
    bundle = ArtifactBundle(processed)
    artifact = TeacherArtifact(root / "teacher_seed_42", bundle, config)
    scores = artifact.scan()
    queries = pl.concat([bundle.scan(f"model_specific/flat/{split}_queries.parquet") for split in ("train", "valid", "test")])
    assert scores.select(pl.len()).collect().item() == queries.select(pl.len()).collect().item()
    assert scores.filter(~pl.col("teacher_available")).select(pl.len()).collect().item() > 0
    for invalid in (scores.limit(1), pl.concat([scores, scores.limit(1)]), scores.with_columns(pl.lit(float("nan")).alias("teacher_score")),
                    scores.with_columns(pl.when(pl.col("teacher_available")).then(pl.col("timestamp")).otherwise(None).alias("teacher_snapshot_ts")),
                    scores.with_columns(pl.lit(.9).alias("teacher_probability"))):
        with pytest.raises(ValueError):
            validate_teacher_rows(invalid, queries, artifact.bank)
    manifest = pl.read_parquet(artifact.manifest_path)
    bad = manifest.with_columns(pl.lit("wrong").alias("sha256"))
    write_batches(artifact.manifest_path, [bad], dict(bad.schema))
    with pytest.raises(ValueError, match="checksum"):
        TeacherArtifact(root / "teacher_seed_42", bundle, config)
    write_batches(artifact.manifest_path, [manifest], dict(manifest.schema))
    with pytest.raises(ValueError, match="smoke"):
        run_distillation(config, "retailrocket", mode=mode, processed_root=processed)
    with pytest.raises(ValueError, match="unknown"):
        run_distillation(config, "retailrocket", mode="wrong", processed_root=processed)


def test_soft_mixture_validation_and_inner_time_guard():
    frame = pl.DataFrame({"target": [0, 1], "teacher_probability": [.2, .8], "timestamp": [10, 10], "x": [1., 2.]})
    assert np.allclose(mixed_targets(frame, .5), [.1, .9])
    for alpha, values in ((1.1, frame), (.5, frame.with_columns(pl.lit(2.).alias("teacher_probability")))):
        with pytest.raises(ValueError):
            mixed_targets(values, alpha)
    with pytest.raises(ValueError, match="distinct"):
        choose_alpha(frame, ["x"], seed=42, smoke=True)
