"""Causal exporters and a complete generator run on adversarial raw fixtures."""

import os
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from src.config.loader import load_config
from src.generator.exporters.graph import graph_at, interaction_edges, iter_metadata_edges
from src.generator.exporters.flat import training_queries, evaluation_queries
from src.generator.exporters.sequences import iter_sequence_samples
from src.generator.metadata import MetadataIndex, asof_item_field
from src.generator.pipeline import generate
from src.generator.quality_gates import (QualityGateError, assert_gates, candidate_gate, graph_gate,
    metadata_gate, negative_gate, prefix_invariance, schema_gate, sequence_gate, temporal_gate, candidate_exclusion_gate)
from src.generator.schemas import EVENTS_SCHEMA, EVENTS_NULLABLE, ITEMS_SCHEMA, SEQUENCES_SCHEMA, EDGES_SCHEMA, STATIC_TS
from src.generator.storage import runtime_workspace, write_batches, write_lazy
from tests.test_evaluation_sampler import facts
from src.generator.evaluation.sampler import iter_evaluation_pools
from src.generator.enrichment.negatives import PrefixState, iter_training_negatives, annotate_future_conflicts
from src.generator.splitting.temporal_splitter import is_primary


def real_events() -> pl.DataFrame:
    """Full-schema events with simultaneous actions and a later repeat."""
    frame = facts("taobao").with_columns(pl.lit(None, dtype=pl.Int64).alias("timestamp_ms"),
        pl.lit("persistent").alias("user_type"), pl.col("event_id").alias("source_row_id"),
        pl.lit(1, dtype=pl.Int64).alias("user_event_seq"), pl.col("timestamp").alias("target_cutoff_ts"))
    return frame.select(EVENTS_SCHEMA.keys()).cast(EVENTS_SCHEMA)


def metadata() -> pl.DataFrame:
    """A changed observed relation and an independent static genre field."""
    rows = [(10, "taobao", "category_id", "2", 10, None, "observed_snapshot", "raw", 10),
            (10, "taobao", "category_id", "3", 20, None, "observed_snapshot", "raw", 20),
            (10, "taobao", "genres", '["Drama","Action"]', STATIC_TS, None, "static_assumed", "raw", None)]
    return pl.DataFrame(rows, schema=ITEMS_SCHEMA, orient="row")


def test_metadata_asof_and_runtime_storage(tmp_path: Path) -> None:
    frame = metadata()
    index = MetadataIndex(frame.lazy(), tmp_path / "metadata.sqlite")
    assert index.category(10, 10) is None
    assert index.category(10, 20) == 2
    assert index.category(10, 21) == 3
    assert list(index.items_in_category(2, 10)) == []
    assert list(index.items_in_category(2, 20)) == [10]
    assert list(index.items_in_category(2, 21)) == []
    assert list(index.items_in_category(3, 21)) == [10]
    assert list(index.items_in_category(-1, 21)) == []
    assert index.get(999, 100, "genres", []) == []
    queries_sent = []
    index.connection.set_trace_callback(queries_sent.append)
    assert index.get(10, 100, "absent_field", []) == []
    assert index.snapshot_timestamp(10, 100, "absent_field") is None
    assert queries_sent == []
    # The field exists in later snapshots: the shortcut cannot future-fill it.
    assert index.category(10, 9) is None and queries_sent
    index.connection.set_trace_callback(None)
    index.close()
    queries = pl.DataFrame({"item_id": [10, 10, 10], "timestamp": [10, 20, 21]}).lazy()
    result = asof_item_field(queries, frame.lazy(), "category_id", pl.Int64).collect()
    assert result["category_id"].to_list() == [None, 2, 3]
    before = os.environ.get("POLARS_OOC_SPILL_DIR")
    with runtime_workspace(tmp_path) as runtime:
        native = Path(os.environ["POLARS_OOC_SPILL_DIR"])
        assert runtime.exists() and native.exists() and native != runtime
        assert native.parent == runtime.parent and os.environ["TMP"] == str(runtime)
    assert not runtime.exists() and os.environ.get("POLARS_OOC_SPILL_DIR") == before
    assert native.exists()
    path = tmp_path / "values.parquet"
    assert write_batches(path, [frame], ITEMS_SCHEMA) == 3
    write_lazy(path, frame.lazy().limit(1))
    assert pl.read_parquet(path).height == 1
    with pytest.raises(ValueError):
        write_batches(tmp_path / "invalid.csv", [], {})
    with pytest.raises(ValueError):
        write_lazy(tmp_path / "invalid.csv", frame.lazy())
    with pytest.raises(ValueError, match="schema"):
        write_batches(path, [pl.DataFrame({"x": [1]})], ITEMS_SCHEMA)
    assert pl.read_parquet(path).height == 1
    assert not path.with_name("values.partial.parquet").exists()


def test_system_temp_storage_is_owned_isolated_and_restored(tmp_path: Path, monkeypatch) -> None:
    import src.generator.storage as storage
    from src.utils.reproducibility import environment_info
    project = tmp_path / "project"
    project.mkdir()
    system_temp = tmp_path / "system_temp"
    monkeypatch.setattr(storage, "SYSTEM_TEMP_ROOT", system_temp)
    monkeypatch.setenv("PIT_RECSYS_RUNTIME_STORAGE", "system_temp")
    keys = ("TMP", "TEMP", "POLARS_TEMP_DIR", "POLARS_OOC_SPILL_DIR")
    original = {key: os.environ.get(key) for key in keys}
    parent = storage.runtime_parent(project)
    assert parent.is_relative_to(system_temp) and not parent.is_relative_to(project)
    assert parent == storage.runtime_parent(project)
    assert parent != storage.runtime_parent(tmp_path / "another_project")
    with pytest.raises(RuntimeError, match="controlled"):
        with storage.runtime_workspace(project) as first:
            native = Path(os.environ["POLARS_OOC_SPILL_DIR"])
            assert native.parent == parent and native != first
            assert os.environ["POLARS_TEMP_DIR"] == str(native)
            assert all(os.environ[key] == str(first) for key in ("TMP", "TEMP"))
            assert environment_info()["runtime_storage_backend"] == "system_temp"
            with storage.runtime_workspace(project) as second:
                assert second != first and second.parent == first.parent == parent
                assert all(os.environ[key] == str(second) for key in ("TMP", "TEMP"))
                assert os.environ["POLARS_OOC_SPILL_DIR"] == str(native)
            assert not second.exists() and first.exists()
            assert all(os.environ[key] == str(first) for key in ("TMP", "TEMP"))
            assert native.exists()
            (first / "owned.sqlite").write_bytes(b"test scratch")
            raise RuntimeError("controlled cleanup")
    assert not first.exists()
    assert native.exists()
    with storage.runtime_workspace(project):
        assert Path(os.environ["POLARS_OOC_SPILL_DIR"]) == native
    assert all(os.environ.get(key) == original[key] for key in keys)
    monkeypatch.setenv("PIT_RECSYS_RUNTIME_STORAGE", "arbitrary_path")
    with pytest.raises(ValueError, match="backend"):
        with storage.runtime_workspace(project):
            pass


@pytest.mark.parametrize("backend", ["project", "system_temp"])
def test_native_spill_survives_consecutive_queries_and_cleans_at_process_exit(tmp_path: Path, backend: str):
    """Exercise real OOC sorts across disposed scopes in a fresh native runtime."""
    program = '''
import json, os, sys
from pathlib import Path
import polars as pl
import src.generator.storage as storage
root = Path(sys.argv[1])
storage.SYSTEM_TEMP_ROOT = root / 'system_temp'
frame = pl.DataFrame({'key': (pl.arange(0, 600000, eager=True) * 7919) % 600000}).with_columns(
    (pl.col('key').cast(pl.String) + pl.lit('x' * 64)).alias('payload'))
paths = []
for step in range(3):
    with storage.runtime_workspace(root) as stage:
        paths.append(str(stage))
        spill = Path(os.environ['POLARS_OOC_SPILL_DIR'])
        if step == 0:
            first_spill = spill
        assert spill == first_spill and spill.exists()
        result = frame.lazy().sort('key', descending=bool(step % 2)).collect(engine='streaming')
        assert result.height == 600000 and result['key'].is_sorted(descending=bool(step % 2))
        assert (spill / str(os.getpid())).is_dir(), 'Native spill directory was not exercised'
    assert not stage.exists() and spill.exists()
print(json.dumps({'native': str(first_spill), 'stages': paths}))
'''
    env = dict(os.environ, PIT_RECSYS_RUNTIME_STORAGE=backend, POLARS_MAX_THREADS="2",
        POLARS_OOC_MEMORY_BUDGET_MB="1", POLARS_OVERRIDE_TOTAL_MEMORY_MB="16")
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-4000:]
    assert "panicked" not in result.stderr and "failed to create spill file" not in result.stderr
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert not Path(receipt["native"]).exists()
    assert all(not Path(path).exists() for path in receipt["stages"])


def test_runtime_parent_rejects_resolved_escape(tmp_path: Path, monkeypatch) -> None:
    import src.generator.storage as storage
    monkeypatch.setenv("PIT_RECSYS_RUNTIME_STORAGE", "project")
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    original = Path.resolve
    def escape(path, *args, **kwargs):
        if path == project / ".venv/runtime":
            return outside
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "resolve", escape)
    with pytest.raises(ValueError, match="escapes"):
        storage.runtime_parent(project)


def test_sequence_graph_and_negative_gates(tmp_path: Path) -> None:
    frame = real_events()
    selected = frame.filter(pl.col("event_id") == 1002).select("event_id").lazy()
    sequences = pl.concat(list(iter_sequence_samples(frame.lazy(), "taobao", "valid", tmp_path / "seq.sqlite", targets=selected, category=lambda item, ts: 2, chunk_rows=1)))
    assert sequences["history_event_ids"].to_list() == [[500, 1000]]
    assert sequences["history_category_ids"].to_list() == [[3, 3]]
    assert all(r.passed for r in sequence_gate(sequences.lazy(), frame.lazy()))
    bad = sequences.with_columns(pl.lit([30, 20], dtype=pl.List(pl.Int64)).alias("history_timestamps"))
    assert not sequence_gate(bad.lazy(), frame.lazy())[0].passed
    with pytest.raises(ValueError, match="registry"):
        list(iter_sequence_samples(frame.lazy(), "taobao", "test", tmp_path / "invalid.sqlite"))
    with pytest.raises(ValueError, match="limit"):
        list(iter_sequence_samples(frame.lazy(), "taobao", "train", tmp_path / "invalid.sqlite", max_history=0))
    with pytest.raises(ValueError, match="wrong split"):
        list(iter_sequence_samples(frame.lazy(), "taobao", "test", tmp_path / "wrong.sqlite", targets=selected))
    assert temporal_gate(frame.lazy()).passed
    assert not temporal_gate(frame.filter(pl.col("split") != "test").lazy()).passed
    assert schema_gate(frame.lazy(), EVENTS_SCHEMA, EVENTS_NULLABLE, ["event_id"], "schema").passed
    assert not schema_gate(frame.drop("label").lazy(), EVENTS_SCHEMA, EVENTS_NULLABLE, ["event_id"], "schema").passed
    assert not schema_gate(pl.concat([frame, frame]).lazy(), EVENTS_SCHEMA, EVENTS_NULLABLE, ["event_id"], "schema").passed
    assert metadata_gate(metadata().lazy()).passed
    assert not metadata_gate(metadata().with_columns(pl.lit("null").alias("field_value")).lazy()).passed
    categories = pl.DataFrame({"category_id": [2], "parent_id": [1]})
    edges = pl.concat([interaction_edges(frame.lazy(), "taobao").collect(), *iter_metadata_edges(metadata().lazy(), categories, "taobao", 1)])
    assert edges.schema == EDGES_SCHEMA and graph_gate(edges.lazy()).passed
    past = graph_at(edges.lazy(), 20).collect()
    assert (past["timestamp"] < 20).all()
    membership = past.filter((pl.col("src_id") == 10) & (pl.col("edge_type") == "belongs_to"))
    assert 2 in membership["dst_id"] and 3 not in membership["dst_id"]
    later = graph_at(edges.lazy(), 21).collect().filter((pl.col("src_type") == "Item") & (pl.col("dst_type") == "Category") & (pl.col("timestamp") != STATIC_TS))
    assert later["dst_id"].to_list() == [3]
    assert not graph_gate(edges.with_columns(pl.lit("Unknown").alias("src_type")).lazy()).passed
    settings = load_config().point_in_time_policy
    train = frame.with_columns(pl.when(pl.col("event_id") == 1000).then(pl.lit("train")).otherwise(pl.col("split")).alias("split"))
    prefix = PrefixState(tmp_path / "negative.sqlite")
    frozen = pl.concat(list(iter_training_negatives(train, settings.synthetic.negatives, is_primary, state=prefix)))
    negatives = annotate_future_conflicts(frozen, prefix)
    prefix.close()
    assert negative_gate(negatives.lazy(), train.lazy()).passed
    assert not negative_gate(negatives.with_columns(pl.lit(1, dtype=pl.Int8).alias("label")).lazy(), train.lazy()).passed
    queries = training_queries(train.lazy(), negatives.lazy(), "taobao").collect()
    assert queries["target"].sum() == 1
    assert not any("conflict" in name for name in queries.columns)
    pools = pl.concat([b.candidates for b in iter_evaluation_pools(frame, "taobao", settings.evaluation, "valid")])
    assert candidate_gate(pools.lazy()).passed
    assert candidate_exclusion_gate(pools.lazy(), frame.lazy(), "taobao", "valid").passed
    corrupt = pools.with_columns(pl.when(pl.col("candidate_label") == 0).then(11).otherwise(pl.col("candidate_item_id")).alias("candidate_item_id"))
    assert not candidate_exclusion_gate(corrupt.lazy(), frame.lazy(), "taobao", "valid").passed
    assert not candidate_exclusion_gate(corrupt.lazy(), frame.lazy(), "yoochoose", "valid", exclude_future=True).passed
    assert not candidate_exclusion_gate(corrupt.lazy(), frame.lazy(), "movielens", "valid").passed
    assert not candidate_gate(pools.head(99).lazy()).passed
    assert evaluation_queries(pools.lazy(), "valid").collect()["target"].sum() == 1
    good = prefix_invariance(lambda f, q: {"n": f.filter(pl.col("timestamp") < q["timestamp"]).height}, frame, {"timestamp": 20})
    bad = prefix_invariance(lambda f, q: {"n": f.height}, frame, {"timestamp": 20})
    assert good.passed and not bad.passed
    assert assert_gates([good]).height == 1
    with pytest.raises(QualityGateError):
        assert_gates([bad])


@pytest.mark.parametrize("personas,collision", [(False, False), (True, False), (False, True)])
def test_complete_raw_generator(tmp_path: Path, personas: bool, collision: bool, monkeypatch) -> None:
    config = load_config(dataset="retailrocket")
    settings = config.dataset("retailrocket")
    raw = tmp_path / "raw"
    raw.mkdir()
    start = 1420070400 + 86350
    rows = [f"{(start+1)*1000},{1000+i},view,{i}," for i in range(1, 131)]
    rows += [f"{(start+10+i)*1000},{1+i%10},transaction,{1+i%20},{i}" for i in range(100)]
    (raw / settings.files["events"]).write_text("timestamp,visitorid,event,itemid,transactionid\n" + "\n".join(rows) + "\n")
    for role in ("properties1", "properties2"):
        (raw / settings.files[role]).write_text(f"timestamp,itemid,property,value\n{start*1000},1,categoryid,2\n")
    (raw / settings.files["categories"]).write_text("categoryid,parentid\n1,\n2,1\n")
    settings = settings.model_copy(update={"raw_dir": raw, "processed_dir": tmp_path / "processed"})
    config = config.model_copy(update={"point_in_time_policy": config.point_in_time_policy.model_copy(update={"datasets": {"retailrocket": settings}})})
    synthetic = config.point_in_time_policy.synthetic
    synthetic = synthetic.model_copy(update={"personas": synthetic.personas.model_copy(update={"enabled": personas, "k": 3})})
    config = config.model_copy(update={"point_in_time_policy": config.point_in_time_policy.model_copy(update={"synthetic": synthetic})})
    isolated = settings.processed_dir / "ablations/fixture" if not personas else None
    if collision:
        import src.generator.enrichment.negatives as negatives_module
        original_seed = negatives_module.stable_seed
        monkeypatch.setattr(negatives_module, "stable_seed", lambda *parts: 42 if parts[0] == "negative" else original_seed(*parts))
        with pytest.raises(QualityGateError, match="negative_schema"):
            generate(config, "retailrocket", output_root=isolated)
        assert pl.read_parquet(isolated / "audit/generation_state.parquet")["status"].item() == "failed"
        assert not pl.read_parquet(isolated / "audit/negative_key_gate.parquet")["passed"].item()
        assert not (isolated / "model_specific/flat/train_queries.parquet").exists()
        assert not (isolated / "audit/artifact_manifest.parquet").exists()
        return
    result = generate(config, "retailrocket", row_limit=1000 if personas else None, output_root=isolated)
    assert result.event_count == 230 and result.valid_groups > 0 and result.test_groups > 0
    assert pl.read_parquet(result.root / "audit/generation_state.parquet")["status"].item() == "complete"
    manifest = pl.read_parquet(result.root / "audit/artifact_manifest.parquet")
    assert "audit/generation_profile.parquet" in manifest["artifact_path"]
    assert all((result.root / path).is_file() for path in manifest["artifact_path"])
    assert pl.read_parquet(result.root / "audit/quality_gates.parquet")["passed"].all()
    if isolated is not None:
        assert result.root == isolated.resolve()
        assert not (settings.processed_dir / "audit/generation_state.parquet").exists()
        for invalid in (tmp_path / "outside", settings.processed_dir / "ablations"):
            with pytest.raises(ValueError, match="distinct child"):
                generate(config, "retailrocket", output_root=invalid)
        with pytest.raises(ValueError, match="full data"):
            generate(config, "retailrocket", row_limit=10, output_root=isolated)
    with pytest.raises(ValueError, match="positive"):
        generate(config, "retailrocket", sample_modulus=0)
    with pytest.raises(ValueError, match="strictly before"):
        generate(config, "retailrocket", row_limit=1)


def test_official_session_isolation(tmp_path: Path) -> None:
    config = load_config(dataset="yoochoose")
    settings = config.dataset("yoochoose")
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = [f"{1000+i},2014-04-01T00:00:00.000Z,{i},1" for i in range(1, 131)]
    rows += ["1,2014-08-31T00:00:00.000Z,10,1", "1,2014-09-01T00:00:00.000Z,20,1", "1,2014-09-16T00:00:00.000Z,30,1"]
    (raw / settings.files["clicks"]).write_text("\n".join(rows) + "\n")
    (raw / settings.files["buys"]).write_text("1,2014-08-31T00:00:00.000Z,10,100,0\n")
    (raw / settings.files["official_test"]).write_text("9000,2014-04-01T00:00:00.000Z,999,1\n9001,2014-04-01T00:00:00.000Z,998,1\n9000,2014-09-17T00:00:00.000Z,31,1\n")
    (raw / settings.files["source_readme"]).write_text("fixture")
    settings = settings.model_copy(update={"raw_dir": raw, "processed_dir": tmp_path / "processed", "include_official_test": True})
    config = config.model_copy(update={"point_in_time_policy": config.point_in_time_policy.model_copy(update={"datasets": {"yoochoose": settings}})})
    result = generate(config, "yoochoose")
    official = result.root / "evaluation/official_click/test"
    candidates = pl.read_parquet(official / "eval_candidates.parquet")
    assert candidates.height == 100 and 998 not in candidates["candidate_item_id"]
    sequences = pl.read_parquet(official / "sequence_samples.parquet")
    assert sequences["history_item_ids"].to_list() == [[999]]
    assert sequences["target_item_id"].to_list() == [31]


def test_genome_requires_explicit_diagnostic(tmp_path: Path) -> None:
    from tests.test_adapters import fixtures
    # The adapter's flag is additionally guarded by the top-level config validator.
    config = load_config(dataset="movielens")
    with pytest.raises(ValueError, match="diagnostic"):
        config.point_in_time_policy.__class__.model_validate(config.point_in_time_policy.model_dump() | {"datasets": {"movielens": config.dataset("movielens").model_copy(update={"genome_scores_enabled": True})}})
    adapter = fixtures(tmp_path, "movielens")
    adapter.settings = adapter.settings.model_copy(update={"genome_scores_enabled": True})
    genome = adapter.metadata_lazy().collect().filter(pl.col("field_name") == "genome_1")
    assert genome["field_value"].to_list() == ["0.9"]
    assert genome["availability_class"].to_list() == ["static_assumed"]
