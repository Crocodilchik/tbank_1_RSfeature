"""Two-Tower full protocol on shared pools, without replacing generator history."""
from pathlib import Path
import polars as pl
from src.models.adv3_seq import pipeline
from src.models.adv3_seq.two_tower_sasrec import SequentialTwoTower
from tests.helpers import model_bundle

def test_sequence_pipeline_real_fit(tmp_path: Path, monkeypatch):
    config, processed = model_bundle(tmp_path)
    monkeypatch.setattr(pipeline, "SequentialTwoTower", lambda **kw: SequentialTwoTower(hidden_size=8, num_layers=kw["num_layers"], attention_heads=kw["attention_heads"], hash_slots=128, dropout=kw["dropout"]))
    root = pipeline.run_sequence(config, "retailrocket", processed_root=processed, smoke=True)
    metrics = pl.read_parquet(root / "seed_42/test_metrics.parquet")
    assert metrics["n_groups"].item() == 5 and metrics["NDCG@10"].is_finite().all()
    predictions = pl.read_parquet(root / "seed_42/test_predictions.parquet")
    assert predictions.height == 500 and predictions["score"].is_finite().all()
    assert pl.read_parquet(root / "run_manifest.parquet")["model"].item() == "adv3"
