"""Atomic typed Parquet outputs and scoped, disposable runtime storage."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterator, Iterable

import polars as pl
import pyarrow.parquet as pq

SYSTEM_TEMP_ROOT = Path(tempfile.gettempdir()).resolve()
_NATIVE_WORKSPACES: dict[Path, tempfile.TemporaryDirectory] = {}


def runtime_parent(project_root: Path) -> Path:
    """Select a contained project cache or an owned system-temp namespace."""
    project = project_root.resolve()
    backend = os.environ.get("PIT_RECSYS_RUNTIME_STORAGE", "project")
    if backend == "project":
        boundary = project
        parent = project / ".venv" / "runtime"
    elif backend == "system_temp":
        boundary = SYSTEM_TEMP_ROOT.resolve()
        identity = hashlib.sha256(os.path.normcase(str(project)).encode("utf-8")).hexdigest()[:16]
        parent = boundary / "pit-recsys" / identity
    else:
        raise ValueError("unknown PIT_RECSYS_RUNTIME_STORAGE backend")
    parent = parent.resolve()
    if parent == boundary or not parent.is_relative_to(boundary):
        raise ValueError("runtime directory escapes its permitted storage root")
    return parent


@contextmanager
def runtime_workspace(project_root: Path) -> Iterator[Path]:
    """Dispose stage files on exit; retain cached native spill paths until process exit."""
    parent = runtime_parent(project_root)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="generator-", dir=parent)
    path = Path(temporary.name).resolve()
    if not path.is_relative_to(parent):
        raise ValueError("temporary directory outside intended runtime root")
    # Polars caches its first OOC directory in process-global native state. A
    # later scope must not delete it while subsequent queries still use it.
    # Retaining the TemporaryDirectory object defers its finalizer to exit.
    if parent not in _NATIVE_WORKSPACES:
        _NATIVE_WORKSPACES[parent] = tempfile.TemporaryDirectory(prefix="native-", dir=parent)
    native = Path(_NATIVE_WORKSPACES[parent].name).resolve()
    if not native.is_relative_to(parent):
        raise ValueError("native spill directory outside intended runtime root")
    old = {key: os.environ.get(key) for key in ("POLARS_TEMP_DIR", "POLARS_OOC_SPILL_DIR", "TMP", "TEMP")}
    os.environ.update(POLARS_TEMP_DIR=str(native), POLARS_OOC_SPILL_DIR=str(native), TMP=str(path), TEMP=str(path))
    try:
        yield path
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        temporary.cleanup()


def write_batches(path: Path, batches: Iterable[pl.DataFrame], schema: dict) -> int:
    """Atomically write bounded typed batches, including correctly typed empty files."""
    if path.suffix != ".parquet":
        raise ValueError("processed artifacts must be Parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.parquet")
    arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
    count = 0
    try:
        with pq.ParquetWriter(partial, arrow_schema, compression="zstd") as writer:
            for frame in batches:
                if set(frame.columns) != set(schema):
                    raise ValueError(f"unexpected batch schema for {path}")
                frame = frame.select(schema.keys()).cast(schema)
                writer.write_table(frame.to_arrow(), row_group_size=100_000)
                count += frame.height
        partial.replace(path)
        return count
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def write_lazy(path: Path, frame: pl.LazyFrame) -> None:
    """Sink a lazy plan atomically without collecting the entire dataset."""
    if path.suffix != ".parquet":
        raise ValueError("processed artifacts must be Parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.stem + ".partial.parquet")
    try:
        frame.sink_parquet(partial, compression="zstd", row_group_size=100_000, engine="streaming")
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
