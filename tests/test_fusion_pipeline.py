"""Actual prefix GraphSAGE fits, OOF provenance and full locked-pool Fusion fit."""
from pathlib import Path
import json
import polars as pl
import pytest
import torch
from src.generator.artifacts import ArtifactBundle
from src.generator.storage import write_batches
from src.models.adv2_gnn.heterogeneous_gnn import HeterogeneousGraphSAGE, TemporalGraphIndex
from src.models.adv3_seq.pipeline import sequence_context
from src.models.adv4_fusion import pipeline
from src.models.adv4_fusion.graph_seq_fusion import GraphSequentialFusion
from src.models.common import temporal_graph
from src.models.common.temporal_graph import CheckpointBank, PrefixGraphEncoder
from src.utils.reproducibility import config_hash, file_hash
from tests.helpers import model_bundle


def test_fusion_pipeline_and_checkpoint_guards(tmp_path: Path, monkeypatch):
    config, processed = model_bundle(tmp_path)
    def small_graph(**kw):
        kw.update(hidden_size=8, hash_slots=128)
        return HeterogeneousGraphSAGE(**kw)
    def small_fusion(**kw):
        kw.update(hidden_size=8, graph_size=8, hash_slots=128)
        return GraphSequentialFusion(**kw)
    monkeypatch.setattr(temporal_graph, "HeterogeneousGraphSAGE", small_graph)
    monkeypatch.setattr(pipeline, "GraphSequentialFusion", small_fusion)
    root = pipeline.run_fusion(config, "retailrocket", processed_root=processed, smoke=True)
    assert pl.read_parquet(root / "run_state.parquet")["status"].item() == "complete"
    assert pl.read_parquet(root / "seed_42/test_metrics.parquet")["n_groups"].item() == 5
    bundle = ArtifactBundle(processed)
    bank_root = root / "upstream/seed_42"
    kwargs = {"expected_config": config_hash(config), "expected_parent": file_hash(bundle.manifest_path)}
    bank = CheckpointBank(bank_root, **kwargs)
    with pytest.raises(ValueError, match="training seed"):
        CheckpointBank(bank_root, expected_seed=43, **kwargs)
    assert len(bank.records) == 4
    for row in bank.records:
        assert row["max_input_ts"] < row["fit_cutoff_ts"] <= row["snapshot_ts"]
        assert row["complete_epochs"] == 2
        assert bank.at(row["snapshot_ts"]) != row
        assert bank.at(row["snapshot_ts"] + 1) == row
    original_bank_hash = file_hash(bank.manifest_path)
    with monkeypatch.context() as patch:
        def refuse_refit(*args, **kw):
            raise AssertionError("An ablation must not refit the shared OOF donor")
        patch.setattr(pipeline, "fit_graph_bank", refuse_refit)
        variant = pipeline.run_fusion(config, "retailrocket", processed_root=processed, smoke=True, graph_enabled=False)
    provenance = json.loads(pl.read_parquet(variant / "run_manifest.parquet")["upstream_provenance"].item())["42"]
    assert provenance["reused"] and provenance["shared_fit_seconds"] > 0
    assert provenance["sha256"] == original_bank_hash and file_hash(bank.manifest_path) == original_bank_hash
    assert pl.read_parquet(variant / "seed_42/test_predictions.parquet").height == 500
    state_path = root / "run_state.parquet"
    original_state = pl.read_parquet(state_path)
    bad_state = original_state.with_columns(pl.lit("running").alias("status"))
    write_batches(state_path, [bad_state], dict(bad_state.schema))
    with pytest.raises(ValueError, match="primary donor"):
        pipeline.run_fusion(config, "retailrocket", processed_root=processed, smoke=True, sequence_enabled=False)
    write_batches(state_path, [original_state], dict(original_state.schema))
    index = TemporalGraphIndex(bundle.graph(), tmp_path / "graph.sqlite")
    encoder = PrefixGraphEncoder(bank, index, torch.device("cpu"), (15, 10))
    assert encoder.encode(1, (1, 2), bank.snapshots[0], -1).snapshot_ts is None
    encoded = encoder.encode(1, (2, 1, 2), 100, -1)
    assert encoded.snapshot_ts == bank.snapshots[-1] and torch.equal(encoded.items[0], encoded.items[2])
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.encode(1, (2, 1, 2), 100, -1) is encoded
    encoder.close()
    runtime = tmp_path / "seed_runtime"
    runtime.mkdir()
    with sequence_context(bundle, runtime) as sequence:
        prepared = pipeline.PreparedFusionBuilder(sequence, config, bundle, index, tmp_path / "extra_bank", torch.device("cpu"), True)
        prepared.prepare_seed(42)
        prepared.prepare_seed(43)
        assert set(prepared.provenance) == {"42", "43"}
        assert prepared.encoder.bank.records[0]["training_seed"] == 43
        prepared.close()
    index.close()
    original = pl.read_parquet(bank.manifest_path)
    def replace(frame):
        write_batches(bank.manifest_path, [frame], dict(frame.schema))
    for column, value in (("max_input_ts", bank.snapshots[-1] + 1), ("checkpoint_path", "../outside.pt")):
        replace(original.with_columns(pl.lit(value).alias(column)))
        with pytest.raises(ValueError):
            CheckpointBank(bank_root, **kwargs)
    replace(original)
    with pytest.raises(ValueError, match="provenance"):
        CheckpointBank(bank_root, expected_config="wrong", expected_parent=kwargs["expected_parent"])
    path = bank.verified_path(bank.records[0])
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        bank.verified_path(bank.records[0])
