"""Past-only graph topology, metadata replacement, cold nodes and real gradients."""

from pathlib import Path
import polars as pl
import pytest
import torch
from torch_geometric.data import Batch

from src.generator.exporters.graph import interaction_edges, iter_metadata_edges
from src.generator.schemas import CATEGORIES_SCHEMA
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE, TemporalGraphIndex, NODE_TYPES, RELATIONS
from src.models.common.embeddings import HashEmbedding
from src.utils.reproducibility import seed_everything
from tests.test_exporters import metadata, real_events


def edges() -> pl.DataFrame:
    """Mixed real interactions and two observed category snapshots."""
    items = metadata().filter(pl.col("field_name") == "category_id")
    return pl.concat([interaction_edges(real_events().lazy(), "taobao").collect(),
        *iter_metadata_edges(items.lazy(), pl.DataFrame(schema=CATEGORIES_SCHEMA), "taobao")])


def test_temporal_graph_sampling_and_cold_forward(tmp_path: Path) -> None:
    seed_everything(42)
    frame = edges()
    index = TemporalGraphIndex(frame.lazy(), tmp_path / "graph.sqlite")
    graph = index.sample(1, [131, 10], 30, target_event_id=1002)
    assert graph.max_edge_ts < 30
    assert {1001, 1002, 2000}.isdisjoint(graph.sampled_edge_ids)
    assert {500, 1000} <= graph.sampled_edge_ids
    assert graph.data["Category"].raw_id.tolist() == [3]
    cold = index.sample(999, [999], 12)
    assert cold.max_edge_ts is None
    model = HeterogeneousGraphSAGE(hidden_size=8, hash_slots=128, dropout=0.)
    model.eval()
    batch = Batch.from_data_list([graph.data, cold.data])
    scores = model(batch)
    assert scores.shape == (3,) and torch.isfinite(scores).all()
    assert torch.allclose(scores[:2], model(graph.data))
    one = model(index.sample(999, [131], 12).data)
    other = model(index.sample(998, [131], 12).data)
    assert torch.allclose(one, other)
    # Future and simultaneous target mutations cannot change sampled inputs.
    mutated = frame.with_columns(pl.when(pl.col("timestamp") >= 30).then(pl.col("dst_id") + 100000).otherwise(pl.col("dst_id")).alias("dst_id"))
    second = TemporalGraphIndex(mutated.lazy(), tmp_path / "mutated.sqlite")
    expected = second.sample(1, [131, 10], 30)
    for kind in NODE_TYPES:
        assert torch.equal(graph.data[kind].raw_id, expected.data[kind].raw_id)
    for relation in RELATIONS:
        assert torch.equal(graph.data[relation].edge_index, expected.data[relation].edge_index)
    second.close()
    tombstone = metadata().head(1).with_columns(pl.lit("-1").alias("field_value"), pl.lit(25).alias("valid_from"), pl.lit(25).alias("snapshot_timestamp"))
    tombstone_edges = pl.concat(list(iter_metadata_edges(tombstone.lazy(), pl.DataFrame(schema=CATEGORIES_SCHEMA), "taobao")))
    third = TemporalGraphIndex(pl.concat([frame, tombstone_edges]).lazy(), tmp_path / "reset.sqlite")
    assert third.sample(1, [10], 30).data["Category"].num_nodes == 0
    third.close()
    with pytest.raises(ValueError, match="target/future"):
        index.sample(1, [10], 30, target_event_id=500)
    for kwargs in ({"user_id": -1}, {"candidate_item_ids": []}, {"candidate_item_ids": [10, 10]}, {"fanout": [0]}):
        with pytest.raises(ValueError):
            index.sample(**({"user_id": 1, "candidate_item_ids": [10], "timestamp": 30} | kwargs))
    index.close()


def test_graph_gradient_on_available_device(tmp_path: Path) -> None:
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = TemporalGraphIndex(edges().lazy(), tmp_path / "graph.sqlite")
    batch = Batch.from_data_list([index.sample(1, [131, 10], 30).data, index.sample(999, [999], 12).data]).to(device)
    model = HeterogeneousGraphSAGE(hidden_size=8, hash_slots=128, dropout=0.).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    target = torch.tensor([1., 0., 0.], device=device)
    before = model(batch).detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch), target)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.embeddings["Item"].tables[0].weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.allclose(before, model(batch))
    index.close()


def test_hash_embeddings_validation_and_missing_ids() -> None:
    encoder = HashEmbedding(8, 128, 2)
    assert torch.equal(encoder(torch.tensor([-1])), torch.zeros(1, 8))
    assert torch.equal(encoder(torch.tensor([10])), encoder(torch.tensor([10])))
    with pytest.raises(ValueError):
        encoder(torch.tensor([1.2]))
    for args in ((0, 128, 1), (8, 7, 1), (8, 15, 1), (8, 128, 3)):
        with pytest.raises(ValueError):
            HashEmbedding(*args)
    with pytest.raises(ValueError):
        HeterogeneousGraphSAGE(num_layers=3)
