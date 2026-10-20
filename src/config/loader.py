"""Typed configuration assembly without implicit overrides or unsafe paths."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from typing_extensions import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DatasetName = Literal["retailrocket", "taobao", "yoochoose", "movielens"]
DATASETS = ("retailrocket", "taobao", "yoochoose", "movielens")


class ConfigurationError(ValueError):
    """Configuration file cannot be assembled safely."""


class StrictModel(BaseModel):
    """Reject unknown fields and implicit scalar coercions."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class TimestampSettings(StrictModel):
    """Unified temporal semantics."""

    unified_unit: Literal["seconds"]
    timezone: Literal["UTC"]
    store_milliseconds_when_available: Literal[True]
    strict_less_than: Literal[True]
    reliable_intra_timestamp_order: Literal[False]


class MetadataSettings(StrictModel):
    """Default metadata join behavior."""

    default_availability: Literal["observed_asof"]
    allow_exact_match_for_static_assumed: Literal[True]


class TargetEncodingSettings(StrictModel):
    """Expanding target encoding hyperparameters."""

    enabled: bool
    smoothing_alpha: float = Field(gt=0)
    min_history_events: int = Field(ge=1)
    empty_prior: float = Field(ge=0, le=1)
    mode: Literal["expanding_point_in_time"]


class GraphSettings(StrictModel):
    """Graph availability rules."""

    snapshot_granularity: Literal["daily", "hourly"]
    allow_exact_match: Literal[False]
    neighbor_sampling_from_snapshot_only: Literal[True]


class SequenceSettings(StrictModel):
    """Common target selection and sequence constraints."""

    max_history_length: int = Field(gt=0, le=50)
    train_strategy: Literal["leave_one_out"]
    eval_strategy: Literal["last_event_only"]
    include_target_event_in_history: Literal[False]
    allow_previous_same_item: Literal["dataset_policy"]
    causal_mask_required: Literal[True]
    empty_history_policy: Literal["keep_with_cold_start"]


class PersonasSettings(StrictModel):
    """Rolling synthetic profile fitting parameters."""

    enabled: bool
    mode: Literal["rolling_snapshot"]
    fit_cutoff_source: Literal["previous_snapshot"]
    k: int = Field(gt=0)
    vector_hash_dim: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    uncertainty_quantiles: list[float]
    min_prior_interactions: int = Field(gt=0)
    mixing_probabilities: list[float]

    @model_validator(mode="after")
    def check_levels(self) -> Self:
        """Require ordered thresholds and three valid mixing probabilities."""
        q = self.uncertainty_quantiles
        if len(q) != 2 or not 0 < q[0] < q[1] < 1:
            raise ValueError("personas requires two ordered quantiles in (0,1)")
        if len(self.mixing_probabilities) != 3 or any(not 0 <= p <= 1 for p in self.mixing_probabilities):
            raise ValueError("personas requires three mixing probabilities in [0,1]")
        return self


class NegativeSettings(StrictModel):
    """Training samples independent of future conflict annotations."""

    enabled: bool
    used_for: Literal["train_only"]
    no_previous_interaction_negatives: Literal[True]
    future_conflict_policy: Literal["keep_as_noise_audit_only"]
    sample_weight: Literal[1.0]
    negatives_per_positive: int = Field(gt=0)
    difficulty_mix: list[float]

    @model_validator(mode="after")
    def check_mix(self) -> Self:
        """Reject invalid difficulty proportions."""
        if len(self.difficulty_mix) != 3 or any(p < 0 for p in self.difficulty_mix) or abs(sum(self.difficulty_mix) - 1) > 1e-9:
            raise ValueError("difficulty_mix must contain three nonnegative probabilities summing to one")
        return self


class SyntheticSettings(StrictModel):
    """Optional enrichment and training sampling settings."""

    personas: PersonasSettings
    negatives: NegativeSettings


class SamplingSettings(StrictModel):
    """Evaluation candidate sampling strategy."""

    strategy: Literal["random", "popularity_biased", "hard"]
    hard_negative_category_depth: int = Field(ge=0)


class EvaluationSettings(StrictModel):
    """One common benchmark contract for every model."""

    candidate_pool_size: Literal[100]
    negatives_per_group: Literal[99]
    seed: int = Field(ge=0)
    target_strategy: Literal["last_event_only"]
    exclude_user_eval_positives: Literal[True]
    exclude_future_positives: bool
    insufficient_candidates: Literal["skip_with_coverage"]
    fallback_mode: Literal["separate_cohort"]
    repeat_purchase_policy: dict[DatasetName, Literal["allow_previous", "allow_within_session_only", "unseen_only"]]
    primary_targets: dict[DatasetName, str]
    fallback_targets: dict[DatasetName, str]
    negative_sampling: SamplingSettings
    catalog_policy: Literal["observed_before_target"]

    @model_validator(mode="after")
    def check_datasets(self) -> Self:
        """Require all datasets and dataset-specific primary targets."""
        for name in ("repeat_purchase_policy", "primary_targets", "fallback_targets"):
            if set(getattr(self, name)) != set(DATASETS):
                raise ValueError(f"{name} must define all four datasets")
        expected = dict(zip(DATASETS, ("transaction", "buy", "click", "rating>=4.0")))
        if self.primary_targets != expected:
            raise ValueError("primary targets must match the approved dataset predicates")
        return self


class AdapterSettings(StrictModel):
    """Raw parser and invalid-record policies."""

    header: bool
    separator: Literal[","]
    encoding: Literal["utf-8"]
    timestamp_unit: Literal["milliseconds", "seconds", "iso8601_utc", "datetime_utc_assumed"]
    invalid_required_field: Literal["error"]
    exact_duplicates: Literal["keep_first"]
    metadata_conflict: Literal["error"]


class DatasetSettings(StrictModel):
    """Dataset-specific sources and explicit metadata assumptions."""

    name: DatasetName
    raw_dir: Path
    processed_dir: Path
    files: dict[str, str]
    adapter: AdapterSettings
    item_properties_availability: Literal["observed_snapshot"] | None = None
    category_hierarchy_availability: Literal["static_assumed"] | None = None
    item_category_availability: Literal["observed_snapshot"] | None = None
    category_id_availability: Literal["static_assumed", "observed_asof"] | None = None
    strict_ablation_category_id_availability: Literal["observed_asof"] | None = None
    user_entity_type: Literal["session"] | None = None
    category_availability: Literal["observed_asof"] | None = None
    primary_eval_mode: Literal["own_temporal_split"] | None = None
    include_official_test: bool | None = None
    official_test_mode: Literal["supplementary_click_only"] | None = None
    official_test_window: Literal["same_as_primary_test"] | None = None
    genres_availability: Literal["static_assumed"] | None = None
    genome_scores_enabled: bool | None = None
    genome_scores_availability: Literal["static_assumed"] | None = None
    genome_scores_usage: Literal["diagnostic_ablation_only"] | None = None
    user_tags_availability: Literal["observed_asof"] | None = None

    @model_validator(mode="after")
    def check_files(self) -> Self:
        """Disallow path injection through raw file basenames."""
        required = {
            "retailrocket": {"item_properties_availability", "category_hierarchy_availability", "item_category_availability"},
            "taobao": {"category_id_availability", "strict_ablation_category_id_availability"},
            "yoochoose": {"user_entity_type", "category_availability", "primary_eval_mode", "include_official_test", "official_test_mode", "official_test_window"},
            "movielens": {"genres_availability", "genome_scores_enabled", "genome_scores_availability", "genome_scores_usage", "user_tags_availability"},
        }
        common = {"name", "raw_dir", "processed_dir", "files", "adapter"}
        if self.model_fields_set != common | required[self.name]:
            raise ValueError(f"missing or inapplicable metadata settings for {self.name}")
        units = dict(zip(DATASETS, ("milliseconds", "seconds", "iso8601_utc", "datetime_utc_assumed")))
        if self.adapter.timestamp_unit != units[self.name] or self.adapter.header != (self.name in ("retailrocket", "movielens")):
            raise ValueError("raw header/time unit contradicts dataset schema")
        expected = {
            "retailrocket": {"events", "properties1", "properties2", "categories"},
            "taobao": {"events"},
            "yoochoose": {"clicks", "buys", "official_test", "source_readme"},
            "movielens": {"ratings", "movies", "links", "tags", "genome_scores", "genome_tags"},
        }
        if set(self.files) != expected[self.name]:
            raise ValueError(f"incorrect source roles for {self.name}")
        if any(not v or Path(v).name != v or "/" in v or "\\" in v or v in (".", "..") for v in self.files.values()):
            raise ValueError("source files must be plain basenames")
        return self


class SplitSettings(StrictModel):
    """Either lower quantiles or exact UTC half-open dates."""

    method: Literal["timestamp_lower_quantile", "fixed_dates"]
    train_quantile: float | None = None
    valid_quantile: float | None = None
    window_start: datetime | None = None
    train_end: datetime | None = None
    valid_end: datetime | None = None
    window_end: datetime | None = None

    @field_validator("window_start", "train_end", "valid_end", "window_end", mode="before")
    @classmethod
    def parse_date(cls, value: Any) -> datetime | None:
        """Parse UTC ISO dates without trusting the machine timezone."""
        if value is None:
            return None
        dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset().total_seconds() != 0:
            raise ValueError("split dates must be timezone-aware UTC")
        return dt.astimezone(timezone.utc)

    @model_validator(mode="after")
    def check_boundaries(self) -> Self:
        """Reject mixed modes, empty windows and reversed boundaries."""
        dates = [self.window_start, self.train_end, self.valid_end, self.window_end]
        if self.method == "timestamp_lower_quantile":
            if any(d is not None for d in dates) or self.train_quantile is None or self.valid_quantile is None or not 0 < self.train_quantile < self.valid_quantile < 1:
                raise ValueError("quantile split requires 0<train_quantile<valid_quantile<1 and no dates")
        elif any(d is None for d in dates) or dates != sorted(dates) or len(set(dates)) != 4 or self.train_quantile is not None or self.valid_quantile is not None:
            raise ValueError("fixed_dates requires four strictly increasing UTC dates and no quantiles")
        return self


class Splits(StrictModel):
    """Split policies for all four datasets."""

    timezone: Literal["UTC"]
    retailrocket: SplitSettings
    taobao: SplitSettings
    yoochoose: SplitSettings
    movielens: SplitSettings


class TrainingSettings(StrictModel):
    """Compute budgets locked before accessing test results."""

    hpo_trials: int = Field(gt=0)
    max_trial_hours: int = Field(gt=0)
    max_neural_epochs: int = Field(gt=0)
    max_booster_iterations: int = Field(gt=0)
    neural_patience: int = Field(gt=0)
    booster_patience: int = Field(gt=0)
    neural_batch_size: int = Field(gt=0)
    gnn_fanout: list[int]
    inner_train_quantiles: list[float]
    valid_hpo_fraction: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def check_folds(self) -> Self:
        """Validate bounded graph sampling and ordered inner folds."""
        q = self.inner_train_quantiles
        if not self.gnn_fanout or any(n <= 0 for n in self.gnn_fanout):
            raise ValueError("gnn_fanout must contain positive bounds")
        if not q or q != sorted(set(q)) or not 0 < q[0] <= q[-1] < 1:
            raise ValueError("inner_train_quantiles must be strictly increasing in (0,1)")
        return self


class ModelSettings(StrictModel):
    """Common closed schema for supported model hyperparameters."""

    backend: Literal["catboost", "lightgbm", "graphsage"] | None = None
    windows_days: list[int] | None = None
    max_feature_depth: int | None = Field(default=None, gt=0, le=2)
    max_candidate_features: int | None = Field(default=None, gt=0)
    selected_features: int | None = Field(default=None, gt=0)
    hidden_size: int | None = Field(default=None, gt=0)
    num_layers: int | None = Field(default=None, gt=0)
    attention_heads: int | None = Field(default=None, gt=0)
    loss: Literal["sampled_bce", "bpr", "contrastive"] | None = None
    graph_feature_mode: Literal["temporal_oof"] | None = None
    lambda_graph: float | None = Field(default=None, ge=0)
    mode: Literal["soft_target_distillation", "score_feature_stacking"] | None = None
    alpha: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def check_dimensions(self) -> Self:
        """Prevent incompatible heads or invalid temporal windows."""
        if self.attention_heads and (not self.hidden_size or self.hidden_size % self.attention_heads):
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.windows_days is not None and (not self.windows_days or any(x <= 0 for x in self.windows_days)):
            raise ValueError("windows_days must be positive")
        if self.selected_features and self.max_candidate_features and self.selected_features > self.max_candidate_features:
            raise ValueError("selected_features exceeds candidate feature limit")
        return self


class RuntimeSettings(StrictModel):
    """Resource settings and reproducible training seeds."""

    seed: int = Field(ge=0, lt=2**32)
    training_seeds: list[int]
    chunk_rows: int = Field(gt=0)
    device: Literal["auto", "cpu", "cuda"]
    deterministic: Literal[True]
    training: TrainingSettings
    models: dict[Literal["baseline", "adv1", "adv2", "adv3", "adv4", "distillation"], ModelSettings]

    @model_validator(mode="after")
    def check_models(self) -> Self:
        """Require the complete approved model roster and unique seeds."""
        if set(self.models) != {"baseline", "adv1", "adv2", "adv3", "adv4", "distillation"}:
            raise ValueError("all five models and distillation must be configured")
        if not self.training_seeds or len(set(self.training_seeds)) != len(self.training_seeds) or min(self.training_seeds) < 0 or max(self.training_seeds) >= 2**32:
            raise ValueError("training_seeds must be unique nonnegative integers")
        return self


class ProjectSettings(StrictModel):
    """Resolved project-owned directories."""

    root: Path
    data_dir: Path
    experiments_dir: Path
    schema_version: Literal["1.0"]


class PolicySettings(StrictModel):
    """Assembled point-in-time rules."""

    version: Literal["1.0"]
    default_mode: Literal["event_level_pit", "diagnostic"]
    timestamp_handling: TimestampSettings
    metadata: MetadataSettings
    target_encoding: TargetEncodingSettings
    graph: GraphSettings
    sequences: SequenceSettings
    synthetic: SyntheticSettings
    evaluation: EvaluationSettings
    datasets: dict[DatasetName, DatasetSettings]

    @model_validator(mode="after")
    def check_strict_metadata(self) -> Self:
        """Disallow undated genome features in strict primary runs."""
        if self.default_mode == "event_level_pit" and any(d.genome_scores_enabled for d in self.datasets.values()):
            raise ValueError("genome_scores requires explicit diagnostic mode")
        return self


class AppConfig(StrictModel):
    """Fully assembled and validated application configuration."""

    project: ProjectSettings
    runtime: RuntimeSettings
    point_in_time_policy: PolicySettings
    splits: Splits

    def dataset(self, name: str) -> DatasetSettings:
        """Return a configured dataset or fail explicitly."""
        try:
            return self.point_in_time_policy.datasets[name]
        except KeyError as exc:
            raise ConfigurationError(f"dataset {name!r} was not loaded") from exc


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting duplicate keys instead of last-wins."""


def _mapping(loader: UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConfigurationError(f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def read_yaml(path: Path) -> dict[str, Any]:
    """Read safe mapping YAML, preserving contextual errors."""
    try:
        value = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path}: expected YAML mapping")
    return value


def _owned_path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ConfigurationError(f"path escapes project root: {value}")
    return path


def load_config(config_dir: Path | str | None = None, dataset: str | None = None) -> AppConfig:
    """Assemble disjoint YAML documents and validate all temporal/path contracts.

    Args:
        config_dir: Config directory; defaults relative to this module, not cwd.
        dataset: Optional single dataset to load, otherwise load all four.
    """
    directory = Path(config_dir).resolve() if config_dir else Path(__file__).resolve().parents[2] / "configs"
    if dataset is not None and dataset not in DATASETS:
        raise ConfigurationError(f"unknown dataset: {dataset}")
    raw = read_yaml(directory / "main_config.yaml")
    if set(raw) != {"project", "runtime", "point_in_time_policy"}:
        raise ConfigurationError("main_config must own exactly project/runtime/point_in_time_policy")
    root = (directory.parent / raw["project"]["root"]).resolve()
    raw["project"]["root"] = root
    for field in ("data_dir", "experiments_dir"):
        raw["project"][field] = _owned_path(root, raw["project"][field])
    for filename, key in (("splits_config.yaml", "splits"), ("evaluation_config.yaml", "evaluation"), ("sequence_config.yaml", "sequences")):
        part = read_yaml(directory / filename)
        if set(part) != {key}:
            raise ConfigurationError(f"{filename} must own exactly {key}")
        destination = raw if key == "splits" else raw["point_in_time_policy"]
        if key in destination:
            raise ConfigurationError(f"duplicate configuration ownership: {key}")
        destination[key] = part[key]
    policy = raw["point_in_time_policy"]
    if "datasets" in policy:
        raise ConfigurationError("main config cannot own datasets")
    policy["datasets"] = {}
    for name in (dataset,) if dataset else DATASETS:
        part = read_yaml(directory / "datasets" / f"{name}.yaml")
        if set(part) != {"dataset"} or part["dataset"].get("name") != name:
            raise ConfigurationError(f"dataset file/name mismatch: {name}")
        ds = part["dataset"]
        for field in ("raw_dir", "processed_dir"):
            ds[field] = _owned_path(root, ds[field])
        expected = raw["project"]["data_dir"] / name
        if ds["raw_dir"] != expected / "raw" or ds["processed_dir"] != expected / "processed":
            raise ConfigurationError("dataset paths must be distinct canonical data/{name}/raw and processed")
        policy["datasets"][name] = ds
    return AppConfig.model_validate(raw)
