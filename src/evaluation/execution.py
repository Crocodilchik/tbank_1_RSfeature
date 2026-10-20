"""Sequential, resumable experiment jobs with process-scoped resource measurements."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import os
import subprocess
import time
from typing import Mapping, Sequence
import polars as pl
import psutil
from filelock import FileLock
from src.generator.storage import write_batches

LOGGER = logging.getLogger(__name__)

JOURNAL_SCHEMA = {"dataset": pl.String, "stage": pl.String, "model": pl.String, "attempt": pl.Int64,
    "status": pl.String, "config_hash": pl.String, "command": pl.String, "started_at_utc": pl.String,
    "finished_at_utc": pl.String, "wall_seconds": pl.Float64, "peak_rss_bytes": pl.Int64,
    "cpu_seconds": pl.Float64, "exit_code": pl.Int64, "log_path": pl.String, "reason": pl.String,
    "measurement": pl.String}


def _publish(path: Path, records: list[dict]) -> None:
    frame = pl.DataFrame(records, schema=JOURNAL_SCHEMA)
    write_batches(path, [frame], JOURNAL_SCHEMA)


def recover_preboot_jobs(journal: Path, *, generation_roots: Mapping[str, Path] | None = None) -> list[dict]:
    """Reconcile attempts proven dead by a host reboot, without terminating processes.

    Resource values remain the last published samples, never inferred final costs.
    Only a running generator state with the identical config hash is invalidated.
    """
    journal.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(journal) + ".lock", timeout=0):
        records = pl.read_parquet(journal).to_dicts() if journal.exists() else []
        return _recover_preboot_jobs(journal, records, generation_roots or {})


def _recover_preboot_jobs(journal: Path, records: list[dict], generation_roots: Mapping[str, Path]) -> list[dict]:
    boot = datetime.fromtimestamp(psutil.boot_time(), timezone.utc)
    recovered = []
    for row in records:
        if row["status"] != "running":
            continue
        started = datetime.fromisoformat(row["started_at_utc"])
        if started.tzinfo is None:
            raise ValueError("running journal attempt requires a timezone-aware start time")
        if started >= boot:
            continue
        reason = ("Host reboot interrupted this attempt; boot=" + boot.isoformat() +
            "; reconciled=" + datetime.now(timezone.utc).isoformat() +
            "; actual exit time/code unknown; resource values are last published samples")
        if row["stage"] == "generator" and row["dataset"] in generation_roots:
            state_path = generation_roots[row["dataset"]] / "audit/generation_state.parquet"
            if state_path.exists():
                state = pl.read_parquet(state_path)
                if state["status"].item() == "running" and state["config_hash"].item() == row["config_hash"]:
                    state = state.with_columns(pl.lit("failed").alias("status"), pl.lit(reason).alias("message"))
                    write_batches(state_path, [state], dict(state.schema))
        row.update(status="interrupted", finished_at_utc=None, exit_code=None, reason=reason,
            measurement=(row["measurement"] or "") + "; pre-boot interruption; final costs unavailable")
        recovered.append(dict(row))
    if recovered:
        _publish(journal, records)
        LOGGER.warning("Recovered %d pre-boot interrupted attempts; sampled costs retained", len(recovered))
    return recovered


def run_job(command: Sequence[str], *, cwd: Path, journal: Path, dataset: str,
            stage: str, model: str, config_hash: str, poll_seconds: float = 1., publish_seconds: float = 30.) -> dict:
    """Run one owned job with exclusive journal ownership and bounded diagnostics."""
    if not command or poll_seconds <= 0 or publish_seconds <= 0:
        raise ValueError("job command and positive polling interval are required")
    journal.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(journal) + ".lock", timeout=0):
        return _run_job(command, cwd=cwd, journal=journal, dataset=dataset, stage=stage,
                        model=model, config_hash=config_hash, poll_seconds=poll_seconds,
                        publish_seconds=publish_seconds)


def _run_job(command: Sequence[str], *, cwd: Path, journal: Path, dataset: str,
             stage: str, model: str, config_hash: str, poll_seconds: float, publish_seconds: float) -> dict:
    """Run an owned subprocess, preserving failed attempts and sampled process-tree costs.

    No row or trial limits are injected. On interruption only this process and
    its children are terminated. Logs are diagnostic output, never instructions.
    """
    records = pl.read_parquet(journal).to_dicts() if journal.exists() else []
    _recover_preboot_jobs(journal, records, {})
    attempt = 1 + max((row["attempt"] for row in records if (row["dataset"], row["stage"], row["model"]) == (dataset, stage, model)), default=0)
    log = journal.parent / "logs" / f"{dataset}_{stage}_{model}_{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    row = {"dataset": dataset, "stage": stage, "model": model, "attempt": attempt, "status": "running",
        "config_hash": config_hash, "command": json.dumps(list(command)), "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None, "wall_seconds": None, "peak_rss_bytes": None, "cpu_seconds": None, "exit_code": None,
        "log_path": str(log), "reason": None, "measurement": "max sampled sum of process-tree RSS; per-process CPU high-water marks"}
    records.append(row)
    _publish(journal, records)
    started = time.perf_counter()
    published = started
    cpu = {}
    with log.open("wb") as output:
        try:
            process = subprocess.Popen(list(command), cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as error:
            row.update(status="failed", reason=str(error), finished_at_utc=datetime.now(timezone.utc).isoformat(),
                wall_seconds=time.perf_counter() - started, peak_rss_bytes=None)
            _publish(journal, records)
            LOGGER.error("cannot start owned job %s/%s: %s", dataset, model, error)
            return row
        try:
            while process.poll() is None:
                rss = 0
                try:
                    parent = psutil.Process(process.pid)
                    children = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        rss += child.memory_info().rss
                        clocks = child.cpu_times()
                        key = (child.pid, child.create_time())
                        cpu[key] = max(cpu.get(key, 0.), clocks.user + clocks.system)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                row["peak_rss_bytes"] = max(row["peak_rss_bytes"] or 0, rss) or None
                if time.perf_counter() - published >= publish_seconds:
                    row["wall_seconds"] = time.perf_counter() - started
                    row["cpu_seconds"] = sum(cpu.values()) if cpu else None
                    _publish(journal, records)
                    published = time.perf_counter()
                time.sleep(poll_seconds)
            row["exit_code"] = process.returncode
            row["status"] = "complete" if process.returncode == 0 else "failed"
        except BaseException:
            row["status"], row["reason"] = "interrupted", "runner interrupted; no result declared"
            if process.poll() is None:
                parent = psutil.Process(process.pid)
                for child in reversed(parent.children(recursive=True)):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait()
            raise
        finally:
            row["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            row["wall_seconds"] = time.perf_counter() - started
            row["cpu_seconds"] = sum(cpu.values()) if cpu else None
            if not row["peak_rss_bytes"]:
                row["peak_rss_bytes"] = None
            _publish(journal, records)
    if row["status"] == "failed":
        with log.open("rb") as diagnostic:
            diagnostic.seek(0, os.SEEK_END)
            diagnostic.seek(max(0, diagnostic.tell() - 8192))
            row["reason"] = diagnostic.read().decode("utf-8", errors="replace")[-2000:]
        _publish(journal, records)
    return row
