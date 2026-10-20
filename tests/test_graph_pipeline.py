"""GraphSAGE end-to-end fit/calibration on the evaluator's unchanged pool."""
from pathlib import Path
import polars as pl
import pytest
from src.models.adv2_gnn import pipeline
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE
from tests.helpers import model_bundle

def test_graph_pipeline_real_fit(tmp_path: Path, monkeypatch):
    config, processed = model_bundle(tmp_path)
    # A small architecture exercises the full algorithm within the unit-test budget.
    monkeypatch.setattr(pipeline, "HeterogeneousGraphSAGE", lambda **kw: HeterogeneousGraphSAGE(hidden_size=8, num_layers=kw["num_layers"], hash_slots=128, dropout=kw["dropout"]))
    root = pipeline.run_graph(config, "retailrocket", processed_root=processed, smoke=True)
    metrics = pl.read_parquet(root / "seed_42/test_metrics.parquet")
    assert metrics["n_groups"].item() == 5 and metrics["NDCG@10"].is_finite().all()
    predictions = pl.read_parquet(root / "seed_42/test_predictions.parquet")
    assert predictions.height == 500 and predictions["score"].is_finite().all()
    assert pl.read_parquet(root / "seed_42/calibration.parquet")["fit_start_ts"].item() > 60
    assert pl.read_parquet(root / "run_manifest.parquet")["mode"].item() == "smoke"
    with pytest.raises(ValueError, match="explicit smoke"):
        pipeline.run_graph(config, "retailrocket", processed_root=processed)


def test_failed_hpo_attempts_are_preserved_before_run_rejection(tmp_path: Path, monkeypatch):
    import src.models.common.pipeline as common
    from src.models.common.training import ResourceLimitError
    config, processed = model_bundle(tmp_path)
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pipeline, "HeterogeneousGraphSAGE", lambda **kw: HeterogeneousGraphSAGE(hidden_size=8, num_layers=kw["num_layers"], hash_slots=128, dropout=kw["dropout"]))
    def exhausted(*args, **kwargs):
        raise ResourceLimitError("controlled fixture budget exhaustion")
    monkeypatch.setattr(common, "fit_neural", exhausted)
    with pytest.raises(ResourceLimitError, match="no adv2 trial"):
        pipeline.run_graph(config, "retailrocket", processed_root=processed, smoke=True)
    root = config.project.experiments_dir / "adv2/retailrocket/smoke"
    attempts = pl.read_parquet(root / "hpo_trials.parquet")
    assert attempts.height == 2 and (attempts["state"] == "FAIL").all()
    assert attempts["NDCG@10"].dtype == pl.Float64 and attempts["NDCG@10"].null_count() == 2
    assert not (root / "run_manifest.parquet").exists()
