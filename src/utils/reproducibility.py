"""Deterministic random streams and auditable artifact provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import psutil

from src.config.loader import AppConfig


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and installed PyTorch without relying on Python hash()."""
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def stable_seed(*parts: object) -> int:
    """Generate the policy-specified big-endian SHA-256 seed."""
    text = "|".join(str(p) for p in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def file_hash(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 with bounded memory."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_value(value: Any, root: Path | None = None) -> Any:
    """Convert supported configuration values into portable JSON values."""
    if isinstance(value, dict):
        return {str(k): canonical_value(v, root) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical_value(v, root) for v in value]
    if isinstance(value, Path):
        return value.relative_to(root).as_posix() if root and value.is_relative_to(root) else value.as_posix()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return value


def config_hash(config: AppConfig) -> str:
    """Hash semantic configuration independently of absolute checkout path."""
    data = config.model_dump(mode="python")
    data["project"].pop("root")
    payload = json.dumps(canonical_value(data, config.project.root), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def environment_info() -> dict[str, Any]:
    """Record runtime, package and resource information, with explicit GPU status."""
    packages = {}
    for name in ("polars", "pyarrow", "pydantic", "numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "catboost", "lightgbm", "optuna", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    memory = psutil.virtual_memory()
    return {"python": sys.version, "platform": platform.platform(), "packages": packages,
            "cpu_threads": os.cpu_count(), "ram_bytes": memory.total, "available_ram_bytes": memory.available,
            "runtime_storage_backend": os.environ.get("PIT_RECSYS_RUNTIME_STORAGE", "project"),
            "system_temp_root": tempfile.gettempdir()}


def code_revision(root: Path) -> str:
    """Return Git HEAD plus dirty status, failing if provenance is unavailable."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if not dirty:
        return head
    digest = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--", "src", "scripts", "configs", "pyproject.toml", "requirements.txt", "requirements-neural.txt"], cwd=root))
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=root, text=True).splitlines()
    for relative in sorted(untracked):
        if relative.startswith(("src/", "scripts/", "configs/")):
            digest.update(relative.encode())
            digest.update(file_hash(root / relative).encode())
    return head + "+dirty." + digest.hexdigest()[:16]


def executed_code_revision() -> str:
    """Identify the imported checkout, including runs using another config/data root."""
    return code_revision(Path(__file__).resolve().parents[2])


def artifact_metadata(config: AppConfig, sources: Iterable[Path], *, fit_cutoff_ts: int | None = None,
                      max_input_ts: int | None = None, snapshot_ts: int | None = None,
                      code_root: Path | None = None) -> dict[str, Any]:
    """Construct artifact provenance and reject temporally impossible fitting."""
    if fit_cutoff_ts is not None and max_input_ts is not None and max_input_ts >= fit_cutoff_ts:
        raise ValueError("artifact fit includes data at/after fit_cutoff_ts")
    if snapshot_ts is not None and fit_cutoff_ts is not None and fit_cutoff_ts > snapshot_ts:
        raise ValueError("fit cutoff must not exceed snapshot timestamp")
    return {"schema_version": config.project.schema_version, "policy_version": config.point_in_time_policy.version,
            "config_hash": config_hash(config), "code_revision": code_revision(code_root or config.project.root),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": config.runtime.seed,
            "fit_cutoff_ts": fit_cutoff_ts, "max_input_ts": max_input_ts, "snapshot_ts": snapshot_ts,
            "mode": config.point_in_time_policy.default_mode,
            "source_hashes": {p.relative_to(config.project.root).as_posix(): file_hash(p) for p in sources},
            "environment": environment_info()}
