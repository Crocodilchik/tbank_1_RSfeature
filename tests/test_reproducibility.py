"""Tests of deterministic sampling streams and truthful artifact lineage."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import pytest

from src.config.loader import load_config
from src.utils.reproducibility import artifact_metadata, canonical_value, code_revision, config_hash, environment_info, file_hash, seed_everything, stable_seed
from src.utils.logging_setup import configure_logging


def test_seed_and_stream_identity() -> None:
    seed_everything(42)
    first = (random.random(), np.random.random())
    seed_everything(42)
    assert (random.random(), np.random.random()) == first
    expected = int.from_bytes(hashlib.sha256(b"42|taobao|primary|test|12|random").digest()[:8], "big") % (2**63 - 1)
    assert stable_seed(42, "taobao", "primary", "test", 12, "random") == expected
    assert stable_seed(42, "a") != stable_seed(43, "a")
    with pytest.raises(ValueError):
        seed_everything(-1)


def test_hashes_and_canonicalization(tmp_path: Path) -> None:
    p = tmp_path / "sample"
    p.write_bytes(b"123456")
    assert file_hash(p, 2) == hashlib.sha256(b"123456").hexdigest()
    assert canonical_value({"b": (p,), "a": datetime(2020, 1, 1, tzinfo=timezone.utc)}, tmp_path) == {"a": "2020-01-01T00:00:00+00:00", "b": ["sample"]}
    assert canonical_value(p) == p.as_posix()
    assert len(config_hash(load_config())) == 64
    with pytest.raises(ValueError):
        file_hash(p, 0)


def test_artifact_fit_guard_and_metadata() -> None:
    config = load_config()
    source = config.project.root / ".gitignore"
    data = artifact_metadata(config, [source], max_input_ts=9, fit_cutoff_ts=10, snapshot_ts=10)
    assert data["source_hashes"][".gitignore"] == file_hash(source)
    assert data["policy_version"] == "1.0"
    assert data["environment"]["packages"]["polars"]
    assert code_revision(config.project.root)
    assert environment_info()["ram_bytes"] > 0
    with pytest.raises(ValueError, match="at/after"):
        artifact_metadata(config, [], max_input_ts=10, fit_cutoff_ts=10)
    with pytest.raises(ValueError, match="snapshot"):
        artifact_metadata(config, [], fit_cutoff_ts=11, snapshot_ts=10)


def test_logging(tmp_path: Path) -> None:
    import logging
    logfile = tmp_path / "logs" / "test.log"
    configure_logging(logfile=logfile)
    logging.getLogger("test").info("sample message")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "sample message" in logfile.read_text(encoding="utf-8")
    configure_logging()
