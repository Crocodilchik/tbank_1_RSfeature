"""Reproducible streaming generation of the common recommendation benchmark."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Iterable
from uuid import uuid4

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from src.config.loader import AppConfig
from src.generator.adapters.registry import make_adapter
from src.generator.enrichment.negatives import PrefixState, annotate_future_conflicts, iter_training_negatives
from src.generator.enrichment.personas import fit_personas
from src.generator.evaluation.sampler import GROUPS_SCHEMA, SKIPS_SCHEMA, iter_evaluation_pools
from src.generator.exporters.flat import QUERY_SCHEMA, evaluation_queries, training_queries
from src.generator.exporters.graph import graph_at, interaction_edges, iter_metadata_edges
from src.generator.exporters.sequences import iter_sequence_samples
from src.generator.metadata import MetadataIndex, asof_item_field
from src.generator.quality_gates import GATES_SCHEMA, GateResult, assert_gates, candidate_gate, candidate_exclusion_gate, graph_gate, metadata_gate, negative_gate, schema_gate, sequence_gate, temporal_gate
from src.generator.schemas import CANDIDATES_NULLABLE, CANDIDATES_SCHEMA, CATEGORIES_SCHEMA, EDGES_NULLABLE, EDGES_SCHEMA, EVENTS_NULLABLE, EVENTS_SCHEMA, ITEMS_NULLABLE, ITEMS_SCHEMA, NEGATIVES_NULLABLE, NEGATIVES_SCHEMA, SEQUENCES_NULLABLE, SEQUENCES_SCHEMA, STATIC_TS, USERS_SCHEMA
from src.generator.splitting.temporal_splitter import assign_splits, is_primary, resolve_boundaries, select_targets
from src.generator.storage import runtime_workspace, write_batches, write_lazy
from src.utils.reproducibility import artifact_metadata, code_revision, config_hash, file_hash

LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA = {"artifact_path": pl.String, "sha256": pl.String, "row_count": pl.Int64, "schema_version": pl.String,
    "policy_version": pl.String, "config_hash": pl.String, "code_revision": pl.String, "created_at_utc": pl.String,
    "seed": pl.Int64, "fit_cutoff_ts": pl.Int64, "max_input_ts": pl.Int64, "snapshot_ts": pl.Int64, "mode": pl.String,
    "source_hashes": pl.String, "environment": pl.String, "parent_artifact_hashes": pl.String}
STATE_SCHEMA = {"status": pl.String, "config_hash": pl.String, "mode": pl.String, "message": pl.String}


@dataclass(frozen=True)
class GenerationResult:
    """Published output location and measured generation counts."""

    root: Path
    mode: str
    event_count: int
    valid_groups: int
    test_groups: int


class ArtifactWriter:
    """Register each completed output, including graph delta segments."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.paths: list[Path] = []

    def lazy(self, relative: str, frame: pl.LazyFrame) -> Path:
        """Write a native streaming query and remember its exact path."""
        path = self.root / relative
        write_lazy(path, frame)
        self.paths.append(path)
        return path

    def batches(self, relative: str, batches: Iterable[pl.DataFrame], schema: dict) -> Path:
        """Write a typed bounded stream and register the completed file."""
        path = self.root / relative
        write_batches(path, batches, schema)
        self.paths.append(path)
        return path

    def frame(self, relative: str, frame: pl.DataFrame) -> Path:
        """Write a small metadata/audit table."""
        return self.batches(relative, [frame], dict(frame.schema))


def _graph_deltas(writer: ArtifactWriter, events: pl.LazyFrame, metadata: pl.LazyFrame,
                  categories: pl.DataFrame, dataset: str, chunk_rows: int) -> list[Path]:
    paths = []
    run_id = uuid4().hex
    batches = iter_metadata_edges(metadata, categories, dataset, chunk_rows)
    import itertools
    for part, batch in enumerate(itertools.chain(interaction_edges(events, dataset).collect_batches(chunk_size=chunk_rows), batches)):
        tagged = batch.with_columns((pl.col("timestamp") // 86400).alias("_day"))
        for group in tagged.partition_by("_day", maintain_order=True):
            name = "base" if group["timestamp"][0] == STATIC_TS else f"day_{group['_day'][0]}"
            relative = f"model_specific/graph/snapshot_id={name}/part-{run_id}-{part:08d}.parquet"
            paths.append(writer.batches(relative, [group.drop("_day")], EDGES_SCHEMA))
    return paths


def _enrich_personas(writer: ArtifactWriter, events: pl.LazyFrame, raw_metadata: pl.LazyFrame,
                     config: AppConfig, dataset: str) -> tuple[list[Path], list[Path], list[tuple[int, np.ndarray]]]:
    settings = config.point_in_time_policy.synthetic.personas
    if not settings.enabled:
        return [], [], []
    timestamps = events.select(pl.col("timestamp").min().alias("minimum"), pl.col("timestamp").max().alias("maximum")).collect().row(0)
    if dataset == "movielens":
        enriched = asof_item_field(events, raw_metadata, "genres", pl.List(pl.String)).with_columns(
            pl.col("genres").fill_null(pl.lit([], dtype=pl.List(pl.String))).list.eval(pl.element().hash(seed=config.runtime.seed).mod(settings.vector_hash_dim).cast(pl.Int64)).alias("category_ids"))
    else:
        enriched = asof_item_field(events, raw_metadata, "category_id", pl.Int64)
    users, items, centroids = [], [], []
    for snapshot_ts in range((timestamps[0] // 86400) * 86400, (timestamps[1] // 86400) * 86400 + 1, 86400):
        for entity in ("user_id", "item_id"):
            LOGGER.info("personas %s snapshot=%s", entity, snapshot_ts)
            snapshot = fit_personas(enriched, snapshot_ts, settings, config.runtime.seed, entity)
            schema = USERS_SCHEMA if entity == "user_id" else ITEMS_SCHEMA
            relative = f"metadata/personas/snapshot_id={snapshot_ts}/{entity}.parquet"
            path = writer.batches(relative, snapshot.iter_metadata(dataset, min(config.runtime.chunk_rows, 100_000)), schema)
            (users if entity == "user_id" else items).append(path)
            if entity == "item_id":
                centroids.append((snapshot_ts, snapshot.centroids.copy()))
                centers = pl.DataFrame({"cluster_id": list(range(1, len(snapshot.centroids) + 1)), "centroid": snapshot.centroids.tolist()},
                                       schema={"cluster_id": pl.Int64, "centroid": pl.List(pl.Float64)})
                writer.frame(f"metadata/personas/snapshot_id={snapshot_ts}/centroids.parquet", centers)
            writer.frame(f"metadata/personas/snapshot_id={snapshot_ts}/{entity}_fit_provenance.parquet", pl.DataFrame({
                "fit_cutoff_ts": [snapshot_ts], "snapshot_ts": [snapshot_ts], "max_input_ts": [snapshot.max_input_ts]},
                schema={"fit_cutoff_ts": pl.Int64, "snapshot_ts": pl.Int64, "max_input_ts": pl.Int64}))
            del snapshot
    return users, items, centroids


def _export_evaluation(writer: ArtifactWriter, events: pl.LazyFrame, config: AppConfig,
                       dataset: str, split: str, temporary: Path, index: MetadataIndex,
                       catalog: pl.LazyFrame, *, mode: str = "primary", fallback: bool = False,
                       catalog_events: pl.LazyFrame | None = None,
                       target_ids: pl.LazyFrame | None = None) -> tuple[int, Path, Path]:
    """Publish one isolated evaluation task, including explicit empty fallback cohorts."""
    key = f"{split}_{mode}"
    prefix = f"evaluation/{split}" if mode == "primary" else f"evaluation/{mode}/{split}"
    fragments = []
    for part, batch in enumerate(iter_evaluation_pools(events, dataset, config.point_in_time_policy.evaluation, split,
            mode=mode, fallback=fallback, database=temporary / f"oracle_{key}.sqlite", metadata_catalog=catalog,
            category=index.category, catalog_events=catalog_events, target_event_ids=target_ids)):
        paths = []
        for name, frame in (("candidates", batch.candidates), ("groups", batch.groups), ("skips", batch.skipped)):
            path = temporary / f"{key}_{name}_{part}.parquet"
            write_batches(path, [frame], dict(frame.schema))
            paths.append(path)
        fragments.append(paths)
    outputs = []
    for i, (name, schema) in enumerate(zip(("eval_candidates", "eval_groups", "skipped_groups"), (CANDIDATES_SCHEMA, GROUPS_SCHEMA, SKIPS_SCHEMA))):
        batches = (frame for fragment in fragments for frame in pl.scan_parquet(fragment[i]).collect_batches(chunk_size=100_000))
        outputs.append(writer.batches(f"{prefix}/{name}.parquet", batches, schema))
    candidates_path, groups_path, _ = outputs
    count = pq.ParquetFile(groups_path).metadata.num_rows
    targets = pl.scan_parquet(groups_path).select(pl.col("eval_group_id").alias("event_id"))
    sequence = writer.batches(f"{prefix}/sequence_samples.parquet", iter_sequence_samples(events, dataset, split,
        temporary / f"sequences_{key}.sqlite", targets=targets,
        max_history=config.point_in_time_policy.sequences.max_history_length, category=index.category), SEQUENCES_SCHEMA)
    flat_name = f"{split}_queries" if mode == "primary" else f"{mode}_{split}_queries"
    writer.lazy(f"model_specific/flat/{flat_name}.parquet", evaluation_queries(pl.scan_parquet(candidates_path), split))
    return count, sequence, candidates_path


def generate(config: AppConfig, dataset: str, *, sample_modulus: int = 1, row_limit: int | None = None,
             output_root: Path | None = None) -> GenerationResult:
    """Generate isolated smoke or full artifacts, refusing invalid/empty benchmarks.

    Raw source files are read-only. All published outputs are Parquet. Temporary
    SQLite/spill indexes live only in an automatically removed runtime workspace.
    """
    if sample_modulus <= 0:
        raise ValueError("sample_modulus must be positive")
    settings = config.dataset(dataset)
    mode = "full" if sample_modulus == 1 and row_limit is None else "smoke"
    root = settings.processed_dir if mode == "full" else settings.processed_dir / "smoke" / f"users_mod_{sample_modulus}_rows_{row_limit or 'all'}"
    if output_root is not None:
        root = output_root.resolve()
        ablations_root = (settings.processed_dir / "ablations").resolve()
        if mode != "full" or root == ablations_root or not root.is_relative_to(ablations_root):
            raise ValueError("explicit output root requires full data and a distinct child of processed/ablations")
    if config.point_in_time_policy.default_mode == "diagnostic":
        root = root / "diagnostic" / config_hash(config)[:12]
        mode = f"diagnostic_{mode}"
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "audit" / "generation_state.parquet"
    digest = config_hash(config)
    code_root = Path(__file__).resolve().parents[2]
    starting_revision = code_revision(code_root)

    def state(status: str, message: str = "") -> None:
        write_batches(state_path, [pl.DataFrame([(status, digest, mode, message)], schema=STATE_SCHEMA, orient="row")], STATE_SCHEMA)

    state("running")
    writer = ArtifactWriter(root)
    started = time.perf_counter()
    try:
        with runtime_workspace(config.project.root) as temporary:
            adapter = make_adapter(settings, row_limit)
            errors = adapter.validate()
            if errors:
                raise ValueError("; ".join(errors))
            adapter.observation_user_modulus = sample_modulus
            observations = adapter.observations_lazy()
            if sample_modulus > 1:
                observations = observations.filter((pl.col("user_id") % sample_modulus) == 0)
            LOGGER.info("normalizing %s mode=%s", dataset, mode)
            normalized = temporary / "normalized.parquet"
            write_lazy(normalized, adapter.normalize_lazy(source=observations))
            boundaries = resolve_boundaries(pl.scan_parquet(normalized), getattr(config.splits, dataset))
            split_events = assign_splits(pl.scan_parquet(normalized), boundaries)
            core_paths = [writer.lazy(f"core/unified_events_{split}.parquet", split_events.filter(pl.col("split") == split)) for split in ("train", "valid", "test")]
            events = pl.scan_parquet(core_paths)
            reference_events = events
            if mode.endswith("smoke"):
                adapter.metadata_item_ids = events.select("item_id").unique()
            categories = adapter.categories()
            writer.frame("metadata/categories.parquet", categories)
            LOGGER.info("exporting item metadata for %s", dataset)
            raw_metadata_path = writer.batches("metadata/items_raw.parquet", adapter.iter_metadata(min(config.runtime.chunk_rows, 100_000)), ITEMS_SCHEMA)
            raw_metadata = pl.scan_parquet(raw_metadata_path)
            user_personas, item_personas, centroid_snapshots = _enrich_personas(writer, events, raw_metadata, config, dataset)
            item_parts = [raw_metadata, *[pl.scan_parquet(p) for p in item_personas]]
            # Hierarchy expansion carries the property's availability timestamp.
            if dataset == "retailrocket":
                lookup = categories.with_columns(pl.col("full_path").str.split("/").list.first().cast(pl.Int64).alias("root_category_id"))
                membership = raw_metadata.filter(pl.col("field_name") == "category_id").with_columns(pl.col("field_value").str.json_decode(pl.Int64).alias("category_id")).join(lookup.lazy().select("category_id", "root_category_id", "level"), on="category_id", how="left")
                for field, source in (("root_category_id", "root_category_id"), ("category_depth", "level")):
                    item_parts.append(membership.filter(pl.col(source).is_not_null()).with_columns(pl.lit(field).alias("field_name"), pl.col(source).cast(pl.String).alias("field_value"), pl.lit("derived").alias("source")).select(ITEMS_SCHEMA.keys()).cast(ITEMS_SCHEMA))
            items_path = writer.lazy("metadata/items.parquet", pl.concat(item_parts))
            writer.batches("metadata/users.parquet", (batch for p in user_personas for batch in pl.scan_parquet(p).collect_batches(chunk_size=100_000)), USERS_SCHEMA)
            index = MetadataIndex(pl.scan_parquet(items_path).filter(pl.col("field_name").is_in(["category_id", "item_cluster"])), temporary / "metadata.sqlite")
            try:
                times = [s for s, _ in centroid_snapshots]
                neighbor_cache: dict[tuple[int, int], set[int]] = {}

                def cluster(item: int, timestamp: int) -> int | None:
                    position = bisect_left(times, timestamp) - 1
                    if position < 0 or index.snapshot_timestamp(item, timestamp, "item_cluster") != times[position]:
                        return None
                    return index.get(item, timestamp, "item_cluster")

                def neighbors(cluster_id: int, timestamp: int) -> set[int]:
                    position = bisect_left(times, timestamp) - 1
                    if position < 0:
                        return set()
                    key = position, cluster_id
                    if key not in neighbor_cache:
                        centers = centroid_snapshots[position][1]
                        if not 0 < cluster_id <= len(centers):
                            return set()
                        ranking = np.argsort(np.square(centers - centers[cluster_id - 1]).sum(axis=1), kind="stable")
                        neighbor_cache[key] = set([int(i) + 1 for i in ranking if i != cluster_id - 1][:2])
                    return neighbor_cache[key]

                def matching_clusters(values: set[int], timestamp: int) -> Iterable[int]:
                    position = bisect_left(times, timestamp) - 1
                    if position < 0:
                        return iter(())
                    return index.items_in_clusters(values, timestamp, times[position])

                LOGGER.info("sampling training negatives %s", dataset)
                prefix = PrefixState(temporary / "training_state.sqlite", negative_id_validation="artifact_gate")
                frozen = temporary / "frozen_negatives.parquet"
                try:
                    write_batches(frozen, iter_training_negatives(events, config.point_in_time_policy.synthetic.negatives, is_primary,
                        config.runtime.seed, prefix, category=index.category, cluster=cluster if centroid_snapshots else None,
                        cluster_neighbors=neighbors if centroid_snapshots else None, chunk_rows=min(config.runtime.chunk_rows, 100_000),
                        category_candidates=index.items_in_category,
                        cluster_candidates=matching_clusters if centroid_snapshots else None), NEGATIVES_SCHEMA)
                    negatives_path = writer.batches("training/negatives.parquet", (annotate_future_conflicts(batch, prefix) for batch in pl.scan_parquet(frozen).collect_batches(chunk_size=100_000)), NEGATIVES_SCHEMA)
                finally:
                    prefix.close()
                negatives = pl.scan_parquet(negatives_path)
                negative_schema_result = schema_gate(negatives, NEGATIVES_SCHEMA, NEGATIVES_NULLABLE,
                    ["negative_id"], "negative_schema")
                writer.frame("audit/negative_key_gate.parquet", pl.DataFrame([asdict(negative_schema_result)], schema=GATES_SCHEMA))
                assert_gates([negative_schema_result])
                LOGGER.info("negative key gate passed; exporting training inputs %s", dataset)
                writer.lazy("model_specific/flat/train_queries.parquet", training_queries(events, negatives, dataset))
                train_ids = negatives.select(pl.col("context_event_id").alias("event_id")).unique()
                sequence_paths = [writer.batches("training/sequence_samples.parquet", iter_sequence_samples(events, dataset, "train", temporary / "sequences_train.sqlite", targets=train_ids,
                    max_history=config.point_in_time_policy.sequences.max_history_length, category=index.category), SEQUENCES_SCHEMA)]
                catalog = pl.scan_parquet(items_path).filter(pl.col("availability_class") != "static_assumed").select("item_id", "valid_from").unique()
                group_counts = {}
                pool_paths = []
                for split in ("valid", "test"):
                    LOGGER.info("generating %s evaluation pools", split)
                    count, sequence, candidates = _export_evaluation(writer, events, config, dataset, split, temporary, index, catalog)
                    group_counts[split] = count
                    if count == 0:
                        raise ValueError(f"no evaluable {split} groups: insufficient catalog/targets; broaden smoke sample")
                    sequence_paths.append(sequence)
                    pool_paths.append(candidates)
                    _, sequence, candidates = _export_evaluation(writer, events, config, dataset, split, temporary, index, catalog, mode="fallback", fallback=True)
                    sequence_paths.append(sequence)
                    pool_paths.append(candidates)
                graph_paths = _graph_deltas(writer, events, pl.scan_parquet(items_path), categories, dataset, 100_000)
                if dataset == "yoochoose" and settings.include_official_test:
                    official_source = adapter.official_observations_lazy()
                    if sample_modulus > 1:
                        official_source = official_source.filter((pl.col("user_id") % sample_modulus) == 0)
                    official = adapter.normalize_lazy(source=official_source).with_columns(
                        pl.when(pl.col("timestamp") < boundaries.train_end).then(pl.lit("train"))
                        .when(pl.col("timestamp") < boundaries.valid_end).then(pl.lit("valid"))
                        .otherwise(pl.lit("test")).alias("split"))
                    official_path = writer.lazy("supplementary/official_session_events.parquet", official)
                    official = pl.scan_parquet(official_path)
                    if official.join(events.select("user_id").unique(), on="user_id", how="semi").limit(1).collect().height:
                        raise ValueError("official sessions overlap primary source sessions")
                    reference_events = pl.concat([events, official])
                    ids = select_targets(official.filter((pl.col("timestamp") >= boundaries.valid_end) & (pl.col("timestamp") < boundaries.window_end)), dataset, "test",
                        config.point_in_time_policy.evaluation.repeat_purchase_policy[dataset]).select("event_id")
                    _, sequence, candidates = _export_evaluation(writer, reference_events, config, dataset, "test", temporary, index,
                        catalog, mode="official_click", catalog_events=events, target_ids=ids)
                    sequence_paths.append(sequence)
                    pool_paths.append(candidates)
                    graph_paths.append(writer.lazy("supplementary/official_context_edges.parquet", interaction_edges(official, dataset)))
            finally:
                index.close()
            LOGGER.info("checking full-artifact structural and PIT gates")
            results = [schema_gate(events, EVENTS_SCHEMA, EVENTS_NULLABLE, ["event_id"], "events_schema"), temporal_gate(events),
                schema_gate(pl.scan_parquet(items_path), ITEMS_SCHEMA, ITEMS_NULLABLE, ["dataset_source", "item_id", "field_name", "valid_from"], "item_schema"), metadata_gate(pl.scan_parquet(items_path)),
                negative_schema_result, negative_gate(negatives, events)]
            for path in sequence_paths:
                samples = pl.scan_parquet(path)
                results.append(schema_gate(samples, SEQUENCES_SCHEMA, SEQUENCES_NULLABLE, ["sample_id"], path.relative_to(root).as_posix()))
                results.extend(sequence_gate(samples, reference_events, config.point_in_time_policy.sequences.max_history_length))
            for path in pool_paths:
                result = pl.scan_parquet(path)
                results.extend([schema_gate(result, CANDIDATES_SCHEMA, CANDIDATES_NULLABLE, ["eval_group_id", "candidate_item_id"], path.relative_to(root).as_posix()), candidate_gate(result)])
                results.append(candidate_exclusion_gate(result, reference_events, dataset, path.parent.name,
                    fallback="fallback" in path.parts, exclude_future=config.point_in_time_policy.evaluation.exclude_future_positives))
            edges = pl.scan_parquet(graph_paths)
            results.extend([schema_gate(edges, EDGES_SCHEMA, EDGES_NULLABLE, ["edge_id"], "graph_schema"), graph_gate(edges)])
            writer.frame("audit/quality_gates.parquet", pl.DataFrame([asdict(r) for r in results], schema=GATES_SCHEMA))
            assert_gates(results)
            event_count = sum(pq.ParquetFile(path).metadata.num_rows for path in core_paths)
            writer.frame("audit/generation_profile.parquet", pl.DataFrame({"dataset": [dataset], "mode": [mode], "event_count": [event_count], "wall_seconds": [time.perf_counter() - started]}))
            sources = [settings.raw_dir / filename for filename in settings.files.values()]
            provenance = artifact_metadata(config, sources, code_root=code_root)
            if provenance["code_revision"] != starting_revision:
                raise ValueError("source code changed during generation; rerun from a fixed revision")
            provenance["mode"] = mode
            manifest = []
            for path in writer.paths:
                row = dict(provenance, artifact_path=path.relative_to(root).as_posix(), sha256=file_hash(path), row_count=pq.ParquetFile(path).metadata.num_rows,
                           parent_artifact_hashes="[]")
                if "personas" in path.parts:
                    snapshot_part = next(part for part in path.parts if part.startswith("snapshot_id="))
                    snapshot_ts = int(snapshot_part.split("=")[1])
                    max_input = events.filter(pl.col("timestamp") < snapshot_ts).select(pl.col("timestamp").max()).collect().item()
                    if max_input is not None and max_input >= snapshot_ts:
                        raise ValueError("persona fit provenance violates strict cutoff")
                    row.update(fit_cutoff_ts=snapshot_ts, snapshot_ts=snapshot_ts, max_input_ts=max_input)
                for key in ("source_hashes", "environment"):
                    row[key] = json.dumps(row[key], sort_keys=True)
                manifest.append(row)
            write_batches(root / "audit/artifact_manifest.parquet", [pl.DataFrame(manifest, schema=MANIFEST_SCHEMA)], MANIFEST_SCHEMA)
        state("complete")
        LOGGER.info("generation complete: %s events=%s valid=%s test=%s", dataset, event_count, group_counts["valid"], group_counts["test"])
        return GenerationResult(root, mode, event_count, group_counts["valid"], group_counts["test"])
    except BaseException as exc:
        state("failed", str(exc))
        LOGGER.exception("generation failed for %s", dataset)
        raise
