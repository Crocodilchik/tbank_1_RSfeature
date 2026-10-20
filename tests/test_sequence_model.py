"""Causal attention, strict historical metadata, empty context and CUDA gradients."""
from pathlib import Path
import copy
import polars as pl
import pytest
import torch

from src.generator.artifacts import ArtifactBundle
from src.generator.metadata import MetadataIndex
from src.models.adv3_seq.two_tower_sasrec import SequenceBatchBuilder, SequenceIndex, SequentialTwoTower
from src.models.adv3_seq.pipeline import sequence_context
from src.models.common.training import iter_query_groups
from src.utils.reproducibility import seed_everything
from tests.helpers import model_bundle
from tests.test_exporters import metadata


def test_sequence_masks_metadata_and_gradient(tmp_path: Path):
    config, root = model_bundle(tmp_path)
    bundle = ArtifactBundle(root)
    groups = list(iter_query_groups(bundle.scan("model_specific/flat/train_queries.parquet")))
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    with sequence_context(bundle, runtime) as builder:
        batch = builder([groups[0], groups[20]])
        assert batch.inputs["history_lengths"].tolist() == [0, 4]
        assert set(batch.inputs).isdisjoint({"target", "target_item_id", "query_group_id", "user_id"})
        seed_everything(42)
        model = SequentialTwoTower(hidden_size=8, hash_slots=128, dropout=0.)
        model.eval()
        states = model.encode_history(batch.inputs)
        mutated = {key: value.clone() for key, value in batch.inputs.items()}
        mutated["history_item_ids"][:, -1] = 999
        changed = model.encode_history(mutated)
        assert torch.allclose(states[:, :-1], changed[:, :-1], atol=1e-6)
        assert torch.allclose(model(batch.inputs)[:groups[0].rows.height], model(mutated)[:groups[0].rows.height], atol=1e-6)
        assert torch.isfinite(model(batch.inputs)).all()
        with pytest.raises(ValueError, match="positional"):
            SequentialTwoTower(hidden_size=8, hash_slots=128, max_history=1)(batch.inputs)
        # Index mutation must fail before a future token reaches attention.
        original = builder.sequences.get(groups[20].group_id)
        corrupted = copy.deepcopy(original)
        corrupted["history_timestamps"][-1] = groups[20].timestamp
        builder.sequences.get.cache_clear()
        import json
        builder.sequences.connection.execute("UPDATE samples SET payload=? WHERE id=?", (json.dumps(corrupted), groups[20].group_id))
        with pytest.raises(ValueError, match="target/future"):
            builder([groups[20]])
        builder.sequences.connection.execute("UPDATE samples SET payload=? WHERE id=?", (json.dumps(original), groups[20].group_id))
        builder.sequences.get.cache_clear()
        # Actual CUDA backward with deterministic math attention.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).train()
        batch = batch.to(device)
        before = model(batch.inputs).detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, foreach=False)
        for _ in range(3):
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(batch.inputs), batch.labels)
            loss.backward()
            assert torch.isfinite(loss) and model.item_tower.item.tables[0].weight.grad.abs().sum() > 0
            optimizer.step()
        assert not torch.allclose(before, model(batch.inputs))
        with pytest.raises(ValueError, match="missing"):
            builder.sequences.get(-999)
        mismatch = copy.copy(groups[0])
        object.__setattr__(mismatch, "timestamp", mismatch.timestamp + 1)
        with pytest.raises(ValueError, match="context"):
            builder([mismatch])
    index = MetadataIndex(metadata().lazy(), tmp_path / "meta.sqlite")
    sequences = SequenceIndex(bundle.scan("training/sequence_samples.parquet"), tmp_path / "seq.sqlite")
    builder = SequenceBatchBuilder(sequences, index)
    assert len(builder.content(10, 10)[0]) == 2  # static genres, exact category unavailable
    assert len(builder.content(10, 20)[0]) == 3 and builder.content(10, 20)[0][0] == 2
    assert builder.content(10, 21)[0][0] == 3
    with pytest.raises(ValueError):
        SequenceBatchBuilder(sequences, index, max_history=51)
    with pytest.raises(ValueError, match="fresh"):
        SequenceIndex(bundle.scan("training/sequence_samples.parquet"), tmp_path / "seq.sqlite")
    builder.close()
    sequences.close()
    index.close()
    for kwargs in ({"hidden_size": 7}, {"attention_heads": 3}, {"num_layers": 3}, {"dropout": 1.}, {"max_history": 51}):
        with pytest.raises(ValueError):
            SequentialTwoTower(**kwargs)
