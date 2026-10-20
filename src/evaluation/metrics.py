"""Exact ranking metrics, fair locked-pool comparison and held-out calibration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

KEYS = ["eval_group_id", "candidate_item_id"]
PREDICTION_SCHEMA = {"eval_group_id": pl.Int64, "candidate_item_id": pl.Int64,
    "score": pl.Float64, "probability": pl.Float64, "model": pl.String, "training_seed": pl.Int64}
DEFAULT_K = (3, 5, 10, 20)


def _vectors(labels: Sequence[int], scores: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or not len(y) or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("expected nonempty finite equal-length vectors with binary labels")
    return y.astype(np.int8), s


def _ranked(labels: Sequence[int], scores: Sequence[float], k: int,
            candidate_ids: Sequence[int] | None) -> tuple[np.ndarray, int]:
    y, s = _vectors(labels, scores)
    if k <= 0 or not isinstance(k, int) or y.sum() == 0:
        raise ValueError("k must be positive and at least one relevant item is required")
    ids = np.arange(len(y)) if candidate_ids is None else np.asarray(candidate_ids)
    if ids.shape != y.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and match labels")
    return y[np.lexsort((ids, -s))][:k], int(y.sum())


def ndcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int,
              candidate_ids: Sequence[int] | None = None) -> float:
    """Compute binary NDCG with score-descending, candidate-ID-ascending ties."""
    ranked, positives = _ranked(labels, scores, k, candidate_ids)
    discount = 1 / np.log2(np.arange(2, len(ranked) + 2))
    ideal = (1 / np.log2(np.arange(2, min(positives, k) + 2))).sum()
    return float(np.dot(ranked, discount) / ideal)


def recall_at_k(labels: Sequence[int], scores: Sequence[float], k: int,
                candidate_ids: Sequence[int] | None = None) -> float:
    """Return the proportion of relevant items appearing in the first K slots."""
    ranked, positives = _ranked(labels, scores, k, candidate_ids)
    return float(ranked.sum() / positives)


def hit_rate_at_k(labels: Sequence[int], scores: Sequence[float], k: int,
                  candidate_ids: Sequence[int] | None = None) -> float:
    """Return one when at least one relevant item appears in the first K slots."""
    ranked, _ = _ranked(labels, scores, k, candidate_ids)
    return float(ranked.any())


def mrr(labels: Sequence[int], scores: Sequence[float], candidate_ids: Sequence[int] | None = None) -> float:
    """Return the untruncated reciprocal rank of the first relevant candidate."""
    ranked, _ = _ranked(labels, scores, len(labels), candidate_ids)
    return float(1 / (np.flatnonzero(ranked)[0] + 1))


def auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Return binary ROC AUC with half credit for tied positive-negative pairs."""
    y, s = _vectors(labels, scores)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined for a one-class group")
    return float((rankdata(s)[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def log_loss(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Compute mean sampled binary log loss, clipping probabilities at 1e-7."""
    y, p = _vectors(labels, probabilities)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean())


def _lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    return frame.lazy() if isinstance(frame, pl.DataFrame) else frame


def per_group_metrics(scored: pl.DataFrame | pl.LazyFrame, ks: Sequence[int] = DEFAULT_K,
                      *, pool_size: int = 100) -> pl.DataFrame:
    """Evaluate single-positive groups lazily, materializing only one result per user.

    Input includes locked candidate_label and user_id supplied by the evaluator.
    `pool_size` defaults to the benchmark's 100; smaller values are for unit fixtures.
    """
    if pool_size < 2 or not ks or any(not isinstance(k, int) or k <= 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("invalid pool size or K values")
    frame = _lazy(scored)
    required = {*KEYS, "user_id", "candidate_label", "score", "probability"}
    if not required <= set(frame.collect_schema()):
        raise ValueError(f"missing scoring columns: {sorted(required - set(frame.collect_schema()))}")
    invalid = frame.filter(pl.any_horizontal(pl.col(c).is_null() for c in required) |
        ~pl.col("score").is_finite() | ~pl.col("probability").is_finite() |
        ~pl.col("probability").is_between(0, 1) | ~pl.col("candidate_label").is_in([0, 1]))
    if invalid.limit(1).collect().height:
        raise ValueError("null/nonfinite scores, invalid probabilities or labels")
    check = frame.group_by("eval_group_id").agg(pl.len().alias("n"), pl.col("candidate_item_id").n_unique().alias("unique"),
        pl.col("candidate_label").sum().alias("positive"), pl.col("user_id").n_unique().alias("users"))
    if check.filter((pl.col("n") != pool_size) | (pl.col("unique") != pool_size) | (pl.col("positive") != 1) | (pl.col("users") != 1)).limit(1).collect().height:
        raise ValueError("groups require the complete unique pool, one user and exactly one positive")
    if frame.select(pl.len()).collect().item() == 0:
        raise ValueError("cannot average an empty cohort; record null with n_groups=0 separately")
    positive = frame.filter(pl.col("candidate_label") == 1).select("eval_group_id", pl.col("score").alias("positive_score"), pl.col("candidate_item_id").alias("positive_id"))
    bound = frame.join(positive, on="eval_group_id")
    is_negative = pl.col("candidate_label") == 0
    p = pl.col("probability").clip(1e-7, 1 - 1e-7)
    rows = bound.group_by("eval_group_id").agg(pl.col("user_id").first(),
        (1 + ((pl.col("score") > pl.col("positive_score")) | ((pl.col("score") == pl.col("positive_score")) & (pl.col("candidate_item_id") < pl.col("positive_id")))).sum()).cast(pl.Int64).alias("rank"),
        (((pl.col("score") < pl.col("positive_score")) & is_negative).sum() + .5 * ((pl.col("score") == pl.col("positive_score")) & is_negative).sum()).truediv(pool_size - 1).alias("AUC"),
        (-(pl.col("candidate_label") * p.log() + (1 - pl.col("candidate_label")) * (1 - p).log())).mean().alias("LogLoss"))
    expressions = [(1 / pl.col("rank")).alias("MRR")]
    for k in ks:
        hit = (pl.col("rank") <= k).cast(pl.Float64)
        expressions.extend([pl.when(pl.col("rank") <= k).then(1 / (pl.col("rank") + 1).log(2)).otherwise(0.).alias(f"NDCG@{k}"),
                            hit.alias(f"Recall@{k}"), hit.alias(f"HitRate@{k}")])
    return rows.with_columns(expressions).sort("eval_group_id").collect(engine="streaming")


def bind_predictions(candidates: pl.DataFrame | pl.LazyFrame,
                     predictions: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """Bind model scores to evaluator-owned answers, rejecting any pool-key mismatch."""
    truth, pred = _lazy(candidates), _lazy(predictions)
    columns = set(pred.collect_schema())
    if not {*KEYS, "score", "probability"} <= columns:
        raise ValueError("prediction keys, score and probability are required")
    pred = pred.select(*KEYS, "score", "probability")
    count, unique = pred.select(pl.len(), pl.struct(KEYS).n_unique().alias("unique")).collect().row(0)
    if count != unique:
        raise ValueError("duplicate prediction keys; select one model and training seed")
    truth_keys = truth.select(KEYS)
    if truth_keys.select(pl.len()).collect().item() != count or truth_keys.join(pred, on=KEYS, how="anti").limit(1).collect().height or pred.join(truth_keys, on=KEYS, how="anti").limit(1).collect().height:
        raise ValueError("prediction keys differ from the locked candidate pool")
    return truth.select(*KEYS, "user_id", "candidate_label").join(pred, on=KEYS)


def pool_fingerprint(candidates: pl.DataFrame | pl.LazyFrame) -> str:
    """Hash canonical locked groups, IDs and evaluator labels with bounded memory."""
    digest = hashlib.sha256()
    frame = _lazy(candidates).select(*KEYS, "candidate_label").sort(KEYS)
    for batch in frame.collect_batches(chunk_size=100_000):
        for row in batch.iter_rows():
            digest.update((json.dumps(row, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def compare_models(candidates: pl.DataFrame | pl.LazyFrame,
                   predictions: Mapping[str, pl.DataFrame | pl.LazyFrame],
                   ks: Sequence[int] = DEFAULT_K, *, pool_size: int = 100) -> pl.DataFrame:
    """Return macro metrics for models evaluated on exactly the same locked pool."""
    if not predictions:
        raise ValueError("at least one model is required")
    fingerprint = pool_fingerprint(candidates)
    results = []
    for model, pred in predictions.items():
        groups = per_group_metrics(bind_predictions(candidates, pred), ks, pool_size=pool_size)
        metrics = [c for c in groups.columns if c not in {"eval_group_id", "user_id", "rank"}]
        record = {"model": model, "n_groups": groups.height, "pool_hash": fingerprint}
        record.update({name: groups[name].mean() for name in metrics})
        results.append(record)
    return pl.DataFrame(results)


@dataclass(frozen=True)
class PlattCalibrator:
    """Monotone sigmoid parameters fitted on late validation only."""

    a: float
    b: float
    fit_start_ts: int
    fit_end_ts: int

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int], timestamps: Sequence[int],
            *, split: str, calibration_start_ts: int) -> PlattCalibrator:
        """Fit sigmoid(a*s+b), a>=0, refusing train/test or early validation rows."""
        y, s = _vectors(labels, scores)
        ts = np.asarray(timestamps)
        if split != "valid" or ts.shape != s.shape or not np.issubdtype(ts.dtype, np.integer) or (ts < calibration_start_ts).any():
            raise ValueError("calibration requires integer late-validation timestamps only")
        if len(np.unique(y)) != 2:
            raise ValueError("one-class calibration is undefined")
        center, scale = float(s.mean()), max(float(s.std()), 1e-12)
        z = (s - center) / scale

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            logits = parameters[0] * z + parameters[1]
            residual = expit(logits) - y
            loss = float(np.mean(np.logaddexp(0, logits) - y * logits))
            return loss, np.array([np.mean(residual * z), np.mean(residual)])

        result = minimize(objective, np.array([1., float(np.log(y.mean() / (1 - y.mean())))]), jac=True,
                          method="L-BFGS-B", bounds=[(0, None), (None, None)], options={"maxiter": 1000})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f"calibration optimizer failed: {result.message}")
        a = float(result.x[0] / scale)
        return cls(a, float(result.x[1] - a * center), int(ts.min()), int(ts.max()))

    @classmethod
    def fit_frame(cls, frame: pl.LazyFrame, *, split: str, calibration_start_ts: int) -> PlattCalibrator:
        """Fit exact monotone Platt parameters by bounded-memory Parquet passes.

        Input has numeric score, binary target and integer timestamp columns.
        The caller should spool joins once before this iterative optimizer.
        """
        if split != "valid" or not frame.collect_schema()["timestamp"].is_integer():
            raise ValueError("calibration requires integer late-validation timestamps only")
        stats = frame.select(pl.len().alias("n"), pl.col("score").mean().alias("mean"),
            pl.col("score").std(ddof=0).alias("std"), pl.col("target").mean().alias("rate"),
            pl.col("timestamp").min().alias("start"), pl.col("timestamp").max().alias("end"),
            (pl.col("score").is_null() | ~pl.col("score").is_finite() | pl.col("target").is_null() |
             ~pl.col("target").is_in([0, 1]) | pl.col("timestamp").is_null()).any().alias("invalid")).collect().row(0, named=True)
        if not stats["n"] or stats["invalid"] or stats["start"] < calibration_start_ts:
            raise ValueError("calibration requires finite scores, binary labels and late-validation timestamps")
        if not 0 < stats["rate"] < 1:
            raise ValueError("one-class calibration is undefined")
        center, scale, count = stats["mean"], max(stats["std"], 1e-12), stats["n"]

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            total = np.zeros(3, dtype=float)
            for batch in frame.select("score", "target").collect_batches(chunk_size=65536):
                z = (batch["score"].to_numpy() - center) / scale
                y = batch["target"].to_numpy()
                logits = parameters[0] * z + parameters[1]
                residual = expit(logits) - y
                total += [np.sum(np.logaddexp(0, logits) - y * logits), np.sum(residual * z), np.sum(residual)]
            return float(total[0] / count), total[1:] / count

        result = minimize(objective, np.array([1., float(np.log(stats["rate"] / (1 - stats["rate"])))]), jac=True,
            method="L-BFGS-B", bounds=[(0, None), (None, None)], options={"maxiter": 1000})
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f"calibration optimizer failed: {result.message}")
        a = float(result.x[0] / scale)
        return cls(a, float(result.x[1] - a * center), int(stats["start"]), int(stats["end"]))

    def predict(self, scores: Sequence[float], *, target_min_ts: int,
                diagnostic_on_fit: bool = False) -> np.ndarray:
        """Calibrate later targets; same-fit results require an explicit diagnostic flag."""
        s = np.asarray(scores, dtype=float)
        if s.ndim != 1 or not np.isfinite(s).all() or (not diagnostic_on_fit and target_min_ts <= self.fit_end_ts):
            raise ValueError("calibration inputs must be finite and strictly after fit timestamps")
        return expit(self.a * s + self.b)


def paired_cluster_bootstrap(groups: pl.DataFrame, left: str, right: str,
                             *, repetitions: int = 2000, seed: int = 42) -> dict[str, float]:
    """Compute a user/session-cluster percentile interval for right-minus-left metric."""
    if repetitions < 2 or groups.is_empty() or groups.select(pl.any_horizontal(pl.col(c).is_null() | ~pl.col(c).is_finite() for c in (left, right)).any()).item():
        raise ValueError("bootstrap requires finite paired values and at least two resamples")
    clusters = groups.sort(["user_id", left, right]).group_by("user_id", maintain_order=True).agg(
        (pl.col(right) - pl.col(left)).sum().alias("total"), pl.len().alias("n"))
    totals, counts = clusters["total"].to_numpy(), clusters["n"].to_numpy()
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        chosen = rng.integers(len(clusters), size=len(clusters))
        differences[index] = totals[chosen].sum() / counts[chosen].sum()
    lo, hi = np.quantile(differences, [.025, .975])
    return {"difference": float(totals.sum() / counts.sum()), "ci_lower": float(lo), "ci_upper": float(hi)}
