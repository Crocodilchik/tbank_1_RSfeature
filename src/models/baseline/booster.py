"""Numeric, explicitly whitelisted GBDT training with temporal provenance."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import polars as pl
import psutil

from src.generator.storage import write_batches

FORBIDDEN = {"target", "label", "candidate_label", "query_group_id", "eval_group_id", "event_id", "target_event_id",
    "candidate_item_id", "user_id", "timestamp", "target_timestamp", "future_positive_conflict", "conflict_first_future_ts", "split"}
CATEGORICAL = {"category_id", "brand_id", "root_category_id"}


class _TimeLimit:
    def __init__(self, seconds: float | None) -> None:
        self.deadline = time.perf_counter() + seconds if seconds is not None else float("inf")

    def after_iteration(self, info: Any) -> bool:
        return time.perf_counter() < self.deadline


def _matrix(frame: pl.DataFrame, features: Sequence[str], backend: str) -> Any:
    if not features or len(set(features)) != len(features) or FORBIDDEN.intersection(features) or any("conflict" in c for c in features):
        raise ValueError("unsafe, duplicate or empty model feature whitelist")
    values = frame.select(features)
    if values.select(pl.any_horizontal(pl.col(c).is_null() | ~pl.col(c).is_finite() for c in features).any()).item():
        raise ValueError("model features must be finite numeric values")
    if backend == "catboost":
        for column in CATEGORICAL.intersection(features):
            if values[column].n_unique() > 65535:
                raise ValueError("categorical cardinality exceeds strict one-hot budget; CTR is forbidden")
        return values.with_columns(pl.col(c).cast(pl.String) for c in CATEGORICAL.intersection(features)).to_pandas()
    return values.to_numpy().astype(np.float32)


@dataclass
class BoosterModel:
    """Fitted booster, train-only feature selection and exact inference contract."""

    estimator: Any
    backend: str
    features: list[str]
    fit_end_ts: int
    training_seed: int

    def predict(self, queries: pl.DataFrame, *, diagnostic_on_train: bool = False) -> np.ndarray:
        """Return raw logits for later queries; historical scoring must be diagnostic."""
        if queries.is_empty():
            return np.empty(0, dtype=float)
        if not diagnostic_on_train and queries["timestamp"].min() <= self.fit_end_ts:
            raise ValueError("prediction target must be strictly after model fit_end_ts")
        matrix = _matrix(queries, self.features, self.backend)
        if self.backend == "catboost":
            return np.asarray(self.estimator.predict(matrix, prediction_type="RawFormulaVal"), dtype=float)
        return np.asarray(self.estimator.predict(matrix, raw_score=True), dtype=float)

    def save(self, directory: Path) -> None:
        """Save native model and a Parquet whitelist/provenance table."""
        directory.mkdir(parents=True, exist_ok=True)
        filename = "model.cbm" if self.backend == "catboost" else "model.txt"
        self.estimator.save_model(str(directory / filename))
        metadata = pl.DataFrame({"backend": [self.backend], "features": [self.features], "fit_end_ts": [self.fit_end_ts], "training_seed": [self.training_seed]})
        write_batches(directory / "model_metadata.parquet", [metadata], dict(metadata.schema))

    @classmethod
    def load(cls, directory: Path) -> BoosterModel:
        """Load the estimator together with its immutable feature whitelist."""
        row = pl.read_parquet(directory / "model_metadata.parquet").row(0, named=True)
        if row["backend"] == "catboost":
            from catboost import CatBoostClassifier
            estimator = CatBoostClassifier()
            estimator.load_model(str(directory / "model.cbm"))
        else:
            import lightgbm as lgb
            estimator = lgb.Booster(model_file=str(directory / "model.txt"))
        return cls(estimator, row["backend"], row["features"], row["fit_end_ts"], row["training_seed"])


def temporal_feature_selection(train: pl.DataFrame, features: Sequence[str], *, seed: int = 42,
                               top_k: int | None = None, already_prefix: bool = False) -> list[str]:
    """Drop constants and select importance on a past inner-train prefix only.

    The last 15% of train time is excluded from selection fitting. The final
    estimator can subsequently fit all train rows using the selected definitions.
    """
    _matrix(train, features, "lightgbm")
    cutoff = train["timestamp"].quantile(.85, interpolation="lower")
    prefix = train if already_prefix else train.filter(pl.col("timestamp") < cutoff)
    active = [name for name in features if prefix[name].n_unique() > 1]
    if not active:
        # A constant feature allows an honest constant-probability baseline.
        return [features[0]]
    if top_k is None or len(active) <= top_k:
        return active
    if top_k <= 0 or prefix["target"].n_unique() != 2:
        raise ValueError("importance selection needs positive top_k and both training classes")
    from catboost import CatBoostClassifier
    selector = CatBoostClassifier(iterations=50, depth=4, random_seed=seed, thread_count=4, verbose=False,
                                 allow_writing_files=False, has_time=True, one_hot_max_size=65535)
    selector.fit(_matrix(prefix, active, "catboost"), prefix["target"].to_numpy(), cat_features=[c for c in active if c in CATEGORICAL])
    importance = selector.feature_importances_
    ranking = sorted(zip(active, importance), key=lambda pair: (-pair[1], pair[0]))
    return [name for name, _ in ranking[:top_k]]


def fit_booster(train: pl.DataFrame, valid: pl.DataFrame | None, features: Sequence[str],
                *, backend: str = "catboost", seed: int = 42, iterations: int = 2000,
                depth: int = 6, learning_rate: float = .05, l2: float = 3., patience: int = 100,
                soft_targets: Sequence[float] | None = None, max_seconds: float | None = None) -> BoosterModel:
    """Fit CatBoost/LightGBM on real train queries and optional sampled soft targets."""
    if backend not in {"catboost", "lightgbm"} or train.is_empty() or iterations <= 0:
        raise ValueError("invalid booster backend, iteration count or empty training data")
    if train["target"].n_unique() != 2 or not train["target"].is_in([0, 1]).all():
        raise ValueError("training requires both binary supervised classes")
    if valid is not None and (valid.is_empty() or train["timestamp"].max() >= valid["timestamp"].min()):
        raise ValueError("validation must be nonempty and strictly after train")
    x = _matrix(train, features, backend)
    y = train["target"].to_numpy() if soft_targets is None else np.asarray(soft_targets, dtype=float)
    if y.shape != (train.height,) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("soft targets must be finite aligned probabilities")
    if backend == "catboost":
        from catboost import CatBoostClassifier, Pool
        cats = [c for c in features if c in CATEGORICAL]
        # One-hot only: CatBoost's internal target statistics are not PIT features.
        estimator = CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2,
            loss_function="CrossEntropy" if soft_targets is not None else "Logloss", random_seed=seed, thread_count=4,
            one_hot_max_size=65535, has_time=True, allow_writing_files=False, verbose=False, used_ram_limit=psutil.Process().memory_info().rss + min(512 * 2**20, int(psutil.virtual_memory().available * .65)))
        training = Pool(x, label=y, cat_features=cats)
        validation = None if valid is None else Pool(_matrix(valid, features, backend), label=valid["target"].to_numpy(), cat_features=cats)
        estimator.fit(training, eval_set=validation, early_stopping_rounds=patience if valid is not None else None, callbacks=[_TimeLimit(max_seconds)])
    else:
        import lightgbm as lgb
        training = lgb.Dataset(x, label=y, feature_name=list(features), free_raw_data=True)
        parameters = {"objective": "cross_entropy", "metric": "binary_logloss", "max_depth": depth, "num_leaves": min(2**depth, 255),
            "learning_rate": learning_rate, "lambda_l2": l2, "seed": seed, "num_threads": 4,
            "deterministic": True, "force_col_wise": True, "verbosity": -1, "min_data_in_leaf": 2}
        validation = None if valid is None else lgb.Dataset(_matrix(valid, features, backend), label=valid["target"].to_numpy(), reference=training)
        deadline = _TimeLimit(max_seconds)
        def stop_on_time(environment: Any) -> None:
            if time.perf_counter() >= deadline.deadline:
                raise lgb.callback.EarlyStopException(environment.iteration, environment.evaluation_result_list or [])
        estimator = lgb.train(parameters, training, num_boost_round=iterations, valid_sets=None if validation is None else [validation],
            callbacks=[stop_on_time] + ([] if validation is None else [lgb.early_stopping(patience, verbose=False)]))
    return BoosterModel(estimator, backend, list(features), int(train["timestamp"].max()), seed)
