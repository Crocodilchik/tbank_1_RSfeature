"""Verified manifest access: stale graph fragments and incomplete runs are excluded."""

from pathlib import Path
import polars as pl
import pyarrow.parquet as pq

from src.utils.reproducibility import file_hash


class ArtifactBundle:
    """A complete generated dataset with immutable, individually verified input files."""

    def __init__(self, root: Path, expected_config_hash: str | None = None) -> None:
        self.root = root.resolve()
        state = pl.read_parquet(self.root / "audit/generation_state.parquet").row(0, named=True)
        if state["status"] != "complete" or (expected_config_hash is not None and state["config_hash"] != expected_config_hash):
            raise ValueError("dataset generation is incomplete or config hash differs; regenerate explicitly")
        self.mode = state["mode"]
        self.config_hash = state["config_hash"]
        self.manifest_path = self.root / "audit/artifact_manifest.parquet"
        manifest = pl.read_parquet(self.manifest_path)
        if manifest["artifact_path"].n_unique() != manifest.height or not (manifest["config_hash"] == self.config_hash).all():
            raise ValueError("invalid artifact manifest keys/config hash")
        self.entries = {row["artifact_path"]: row for row in manifest.iter_rows(named=True)}
        self.verified: set[str] = set()

    def path(self, relative: str) -> Path:
        """Resolve and verify one exact manifest entry before model consumption."""
        if relative not in self.entries:
            raise ValueError(f"artifact is absent from current manifest: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("artifact escapes dataset root")
        if relative not in self.verified:
            entry = self.entries[relative]
            if file_hash(path) != entry["sha256"] or pq.ParquetFile(path).metadata.num_rows != entry["row_count"]:
                raise ValueError(f"artifact checksum/row count mismatch: {relative}")
            self.verified.add(relative)
        return path

    def scan(self, relative: str) -> pl.LazyFrame:
        """Read a validated artifact lazily."""
        return pl.scan_parquet(self.path(relative))

    def events(self) -> pl.LazyFrame:
        """Replay exactly the three primary real-event partitions."""
        return pl.concat([self.scan(f"core/unified_events_{split}.parquet") for split in ("train", "valid", "test")])

    def graph(self) -> pl.LazyFrame:
        """Read only graph parts registered in this completed generator run."""
        paths = [self.path(name) for name in self.entries if name.startswith("model_specific/graph/")]
        if not paths:
            raise ValueError("no graph parts in manifest")
        return pl.scan_parquet(paths)
