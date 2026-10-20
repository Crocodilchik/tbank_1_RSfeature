"""Physical schemas synchronized with approved PIT policy section 9."""

import polars as pl

STATIC_TS = -(2**63)

EVENTS_SCHEMA = {
    "event_id": pl.Int64,
    "user_id": pl.Int64,
    "item_id": pl.Int64,
    "timestamp": pl.Int64,
    "timestamp_ms": pl.Int64,
    "label": pl.Int8,
    "event_type": pl.String,
    "dataset_source": pl.String,
    "user_type": pl.String,
    "is_synthetic": pl.Boolean,
    "source_row_id": pl.Int64,
    "user_event_seq": pl.Int64,
    "split": pl.String,
    "target_cutoff_ts": pl.Int64,
}
EVENTS_NULLABLE = {'timestamp_ms'}

ITEMS_SCHEMA = {
    "item_id": pl.Int64,
    "dataset_source": pl.String,
    "field_name": pl.String,
    "field_value": pl.String,
    "valid_from": pl.Int64,
    "valid_to": pl.Int64,
    "availability_class": pl.String,
    "source": pl.String,
    "snapshot_timestamp": pl.Int64,
}
ITEMS_NULLABLE = {'valid_to', 'snapshot_timestamp'}

USERS_SCHEMA = {
    "user_id": pl.Int64,
    "dataset_source": pl.String,
    "field_name": pl.String,
    "field_value": pl.String,
    "valid_from": pl.Int64,
    "valid_to": pl.Int64,
    "availability_class": pl.String,
    "feature_source": pl.String,
    "fit_cutoff_ts": pl.Int64,
    "cluster_level": pl.String,
    "seed": pl.Int64,
}
USERS_NULLABLE = {'fit_cutoff_ts', 'seed', 'valid_to', 'cluster_level'}

EDGES_SCHEMA = {
    "edge_id": pl.Int64,
    "src_type": pl.String,
    "src_id": pl.Int64,
    "dst_type": pl.String,
    "dst_id": pl.Int64,
    "edge_type": pl.String,
    "timestamp": pl.Int64,
    "valid_from": pl.Int64,
    "valid_to": pl.Int64,
    "weight": pl.Float64,
    "dataset_source": pl.String,
    "split_snapshot_id": pl.String,
    "source": pl.String,
}
EDGES_NULLABLE = {'valid_to', 'weight'}

SEQUENCES_SCHEMA = {
    "sample_id": pl.Int64,
    "user_id": pl.Int64,
    "target_event_id": pl.Int64,
    "target_item_id": pl.Int64,
    "target_timestamp": pl.Int64,
    "target_label": pl.Int8,
    "cutoff_ts": pl.Int64,
    "history_event_ids": pl.List(pl.Int64),
    "history_item_ids": pl.List(pl.Int64),
    "history_timestamps": pl.List(pl.Int64),
    "history_labels": pl.List(pl.Int8),
    "history_category_ids": pl.List(pl.Int64),
    "history_time_deltas": pl.List(pl.Int64),
    "max_history_timestamp": pl.Int64,
    "dataset_source": pl.String,
    "split": pl.String,
}
SEQUENCES_NULLABLE = {'max_history_timestamp'}

NEGATIVES_SCHEMA = {
    "negative_id": pl.Int64,
    "user_id": pl.Int64,
    "item_id": pl.Int64,
    "context_event_id": pl.Int64,
    "context_timestamp": pl.Int64,
    "timestamp": pl.Int64,
    "label": pl.Int8,
    "negative_type": pl.String,
    "dataset_source": pl.String,
    "sampling_seed": pl.Int64,
    "catalog_snapshot_ts": pl.Int64,
    "future_positive_conflict": pl.Boolean,
    "conflict_first_future_ts": pl.Int64,
}
NEGATIVES_NULLABLE = {'conflict_first_future_ts'}

CANDIDATES_SCHEMA = {
    "eval_group_id": pl.Int64,
    "user_id": pl.Int64,
    "target_event_id": pl.Int64,
    "target_timestamp": pl.Int64,
    "ground_truth_item_id": pl.Int64,
    "candidate_item_id": pl.Int64,
    "candidate_label": pl.Int8,
    "candidate_type": pl.String,
    "dataset_source": pl.String,
    "sampling_seed": pl.Int64,
    "catalog_snapshot_ts": pl.Int64,
    "exclusion_policy": pl.String,
}
CANDIDATES_NULLABLE = set()

CATEGORIES_SCHEMA = {"category_id": pl.Int64, "dataset_source": pl.String, "parent_id": pl.Int64, "level": pl.Int8, "name": pl.String, "full_path": pl.String}
