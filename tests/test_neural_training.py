"""Complete-group training, deadline handling, exact-pool scoring and save/restore."""
from pathlib import Path
import polars as pl
import pytest
import torch
from torch import nn

from src.generator.exporters.flat import QUERY_SCHEMA
from src.models.common.training import NeuralBatch, ResourceLimitError, fit_neural, group_batches, iter_query_groups, predict_batches
from tests.helpers import model_bundle

class LinearScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    def forward(self, inputs):
        return self.layer(inputs["x"]).squeeze(-1)

def builder(groups):
    rows = pl.concat([group.rows for group in groups])
    return NeuralBatch({"x": torch.tensor(rows["candidate_item_id"].to_numpy(), dtype=torch.float32).unsqueeze(-1) / 100.},
        torch.tensor(rows["target"].to_numpy(), dtype=torch.float32), rows)

def test_complete_epochs_restore_and_locked_pools(tmp_path: Path):
    _, root = model_bundle(tmp_path)
    train = pl.scan_parquet(root / "model_specific/flat/train_queries.parquet")
    valid = pl.scan_parquet(root / "model_specific/flat/valid_queries.parquet").sort(["timestamp", "query_group_id", "candidate_item_id"])
    truth = pl.scan_parquet(root / "evaluation/valid/eval_candidates.parquet")
    device = torch.device("cpu")
    model = LinearScorer()
    result = fit_neural(model, builder, train, valid, truth, tmp_path / "best.pt", model_name="tiny", seed=42,
        device=device, epochs=3, batch_groups=10, patience=1)
    assert result.complete_epochs >= 2 and result.best_epoch >= 1
    assert result.train_rows == train.select(pl.len()).collect().item()
    restored = LinearScorer()
    restored.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=True))
    a = pl.concat(list(predict_batches(model, builder, valid, model_name="tiny", seed=42, device=device, fit_end_ts=result.fit_end_ts)))
    b = pl.concat(list(predict_batches(restored, builder, valid, model_name="tiny", seed=42, device=device, fit_end_ts=result.fit_end_ts, probabilities=lambda x: x * 0 + .5)))
    assert a["score"].equals(b["score"]) and (b["probability"] == .5).all()
    assert a.height == valid.select(pl.len()).collect().item()
    last = fit_neural(model, builder, train, None, None, tmp_path / "last.pt", model_name="tiny", seed=42,
        device=device, epochs=2, batch_groups=11)
    assert last.best_epoch == 2 and last.best_ndcg is None
    with pytest.raises(ResourceLimitError, match="one full"):
        fit_neural(model, builder, train, None, None, tmp_path / "fail.pt", model_name="tiny", seed=42, device=device, epochs=1, max_seconds=0)
    for options in ({"epochs": 0}, {"batch_groups": 0}, {"patience": 0}, {"valid": valid, "valid_truth": None}, {"train": train.limit(0)}, {"valid": train, "valid_truth": truth}, {"valid": valid.limit(0), "valid_truth": truth}):
        arguments = dict(model=model, builder=builder, train=train, valid=None, valid_truth=None, checkpoint=tmp_path / "bad.pt", model_name="tiny", seed=42, device=device, epochs=1)
        arguments.update(options)
        with pytest.raises(ValueError):
            fit_neural(**arguments)
    with pytest.raises(ValueError, match="strictly after"):
        list(predict_batches(model, builder, valid, model_name="tiny", seed=42, device=device, fit_end_ts=1000))
    with pytest.raises(ValueError, match="misaligned/nonfinite"):
        list(predict_batches(nn.Identity(), builder, valid, model_name="tiny", seed=42, device=device, fit_end_ts=0))

def test_group_validation_across_batches():
    # 8300 rows force a query group to cross the reader's 8192-row boundary.
    rows = [(group, group, candidate, group, int(candidate == 0), "train", "retailrocket") for group in range(83) for candidate in range(100)]
    queries = pl.DataFrame(rows, schema=QUERY_SCHEMA, orient="row").lazy()
    assert len(list(iter_query_groups(queries, expected_size=100))) == 83
    assert sum(len(batch) for batch in group_batches(queries, 9)) == 83
    for invalid in (queries.sort("candidate_item_id"), queries.with_columns(pl.lit(0).cast(pl.Int8).alias("target")), queries.limit(99)):
        with pytest.raises(ValueError):
            list(iter_query_groups(invalid, expected_size=100))
    with pytest.raises(ValueError):
        list(group_batches(queries, 0))
