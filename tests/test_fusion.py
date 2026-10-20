"""Fusion component isolation and gradient flow without teacher-weight updates."""
from pathlib import Path
from types import SimpleNamespace
import polars as pl
import pytest
import torch
from src.generator.artifacts import ArtifactBundle
from src.models.adv3_seq.pipeline import sequence_context
from src.models.adv4_fusion.graph_seq_fusion import FusionBatchBuilder, GraphSequentialFusion
from src.models.common.temporal_graph import GraphEncoding
from src.models.common.training import iter_query_groups
from tests.helpers import model_bundle

class PrefixFixture:
    width = 8
    def encode(self, user, items, timestamp, target):
        return GraphEncoding(torch.arange(8).float(), torch.arange(len(items) * 8).reshape(-1, 8).float() / 100., torch.zeros(len(items)), timestamp - 1)

def test_fusion_components_and_backward(tmp_path: Path):
    _, root = model_bundle(tmp_path)
    bundle = ArtifactBundle(root)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    groups = list(iter_query_groups(bundle.scan("model_specific/flat/train_queries.parquet")))
    with sequence_context(bundle, runtime) as sequence:
        builder = FusionBatchBuilder(sequence, PrefixFixture())
        batch = builder([groups[0], groups[20]])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(device)
        model = GraphSequentialFusion(hidden_size=8, graph_size=8, hash_slots=128, dropout=0., lambda_graph=.1).to(device)
        logits = model(batch.inputs)
        loss = model.training_loss(batch.inputs, batch.labels, logits)
        loss.backward()
        assert torch.isfinite(loss) and model.history_fusion.weight.grad.abs().sum() > 0
        assert model.graph_item_projection.weight.grad.abs().sum() > 0
        for flag in ("graph_enabled", "sequence_enabled", "metadata_enabled"):
            component = GraphSequentialFusion(hidden_size=8, graph_size=8, hash_slots=128, dropout=0., **{flag: False}).to(device).eval()
            inputs = {key: value.clone() for key, value in batch.inputs.items()}
            changed_keys = [key for key in inputs if key.startswith("graph_")] if flag == "graph_enabled" else ["history_item_ids", "history_labels"] if flag == "sequence_enabled" else [key for key in inputs if key.endswith("category_ids") or key.endswith("brand_ids")]
            for key in changed_keys:
                inputs[key] = inputs[key] + (1 if key == "history_labels" else 100)
            assert torch.allclose(component(batch.inputs), component(inputs), atol=1e-6)
            assert torch.isfinite(component.training_loss(batch.inputs, batch.labels, component(batch.inputs)))
        bad = PrefixFixture()
        bad.encode = lambda user, items, timestamp, target: GraphEncoding(torch.zeros(8), torch.zeros(len(items), 8), torch.zeros(len(items)), timestamp)
        with pytest.raises(ValueError, match="snapshot"):
            FusionBatchBuilder(sequence, bad)([groups[0]])
    for options in ({"graph_size": 0}, {"lambda_graph": .5}):
        with pytest.raises(ValueError):
            GraphSequentialFusion(hidden_size=8, hash_slots=128, **options)
