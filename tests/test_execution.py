"""Owned subprocess jobs preserve success, failure, interruption and resource lineage."""
from pathlib import Path
import sys
import polars as pl
import pytest
from filelock import FileLock, Timeout
from src.evaluation.execution import JOURNAL_SCHEMA, recover_preboot_jobs, run_job
from src.generator.storage import write_batches


def test_job_success_failure_and_spawn_error(tmp_path: Path):
    arguments = dict(cwd=tmp_path, journal=tmp_path / "journal.parquet", dataset="fixture", stage="model", model="fixture", config_hash="fixture", poll_seconds=.02)
    result = run_job([sys.executable, "-c", "import time;print('fixture success');time.sleep(.3)"], **arguments)
    assert result["status"] == "complete" and result["exit_code"] == 0 and result["peak_rss_bytes"] > 0
    assert result["wall_seconds"] > .2 and "fixture success" in Path(result["log_path"]).read_text()
    result = run_job([sys.executable, "-c", "import sys;print('fixture failure');sys.exit(7)"], **arguments)
    assert result["status"] == "failed" and result["exit_code"] == 7 and result["attempt"] == 2
    assert "fixture failure" in result["reason"]
    result = run_job([str(tmp_path / "missing-executable")], **arguments)
    assert result["status"] == "failed" and result["exit_code"] is None
    assert pl.read_parquet(arguments["journal"]).height == 3
    with pytest.raises(ValueError):
        run_job([], **arguments)


def test_job_interruption_terminates_only_owned_tree(tmp_path: Path, monkeypatch):
    import src.evaluation.execution as execution
    def interrupt(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(execution.time, "sleep", interrupt)
    journal = tmp_path / "journal.parquet"
    with pytest.raises(KeyboardInterrupt):
        run_job([sys.executable, "-c", "import time;time.sleep(5)"], cwd=tmp_path, journal=journal,
            dataset="fixture", stage="model", model="fixture", config_hash="fixture")
    assert pl.read_parquet(journal)["status"].item() == "interrupted"


def test_live_journal_contains_measured_progress(tmp_path: Path, monkeypatch):
    import src.evaluation.execution as execution
    publish = execution._publish
    snapshots = []
    def capture(path, records):
        snapshots.append(dict(records[-1]))
        publish(path, records)
    monkeypatch.setattr(execution, "_publish", capture)
    execution.run_job([sys.executable, "-c", "import time;time.sleep(.3)"], cwd=tmp_path,
        journal=tmp_path / "journal.parquet", dataset="fixture", stage="model", model="fixture",
        config_hash="fixture", poll_seconds=.02, publish_seconds=.03)
    assert snapshots[0]["peak_rss_bytes"] is None and snapshots[0]["cpu_seconds"] is None
    measured = [row for row in snapshots if row["status"] == "running" and row["wall_seconds"] is not None]
    assert measured and all(row["peak_rss_bytes"] > 0 and row["cpu_seconds"] >= 0 for row in measured)
    assert snapshots[-1]["status"] == "complete"


def test_concurrent_runner_cannot_replace_journal(tmp_path: Path):
    journal = tmp_path / "journal.parquet"
    with FileLock(str(journal) + ".lock"):
        with pytest.raises(Timeout):
            run_job([sys.executable, "-c", "pass"], cwd=tmp_path, journal=journal,
                dataset="fixture", stage="model", model="fixture", config_hash="fixture")
    assert not journal.exists()


def interrupted_fixture(tmp_path, monkeypatch, *, state_status="running", state_hash="fixture"):
    import src.evaluation.execution as execution
    monkeypatch.setattr(execution.psutil, "boot_time", lambda: 1704067200.)  # 2024-01-01 UTC
    row = dict.fromkeys(JOURNAL_SCHEMA)
    row.update(dataset="fixture", stage="generator", model="all", attempt=1, status="running",
        config_hash="fixture", command="[]", started_at_utc="2023-12-31T23:00:00+00:00",
        wall_seconds=123., cpu_seconds=87., peak_rss_bytes=4096, measurement="sampled process tree")
    journal = tmp_path / "journal.parquet"
    write_batches(journal, [pl.DataFrame([row], schema=JOURNAL_SCHEMA)], JOURNAL_SCHEMA)
    root = tmp_path / "processed"
    state = pl.DataFrame({"status": [state_status], "config_hash": [state_hash], "mode": ["full"], "message": [""]})
    write_batches(root / "audit/generation_state.parquet", [state], dict(state.schema))
    return journal, root, row


@pytest.mark.parametrize("state_status,state_hash,expected", [
    ("running", "fixture", "failed"), ("complete", "fixture", "complete"),
    ("failed", "fixture", "failed"), ("running", "other", "running")])
def test_reboot_recovery_retains_samples_and_guards_generation(tmp_path, monkeypatch, state_status, state_hash, expected):
    journal, root, original = interrupted_fixture(tmp_path, monkeypatch, state_status=state_status, state_hash=state_hash)
    recovered = recover_preboot_jobs(journal, generation_roots={"fixture": root})
    assert len(recovered) == 1 and recovered[0]["status"] == "interrupted"
    assert recovered[0]["finished_at_utc"] is None and recovered[0]["exit_code"] is None
    assert "reboot" in recovered[0]["reason"] and "last published" in recovered[0]["reason"]
    for column in ("wall_seconds", "cpu_seconds", "peak_rss_bytes", "started_at_utc", "attempt", "command"):
        assert recovered[0][column] == original[column]
    assert pl.read_parquet(root / "audit/generation_state.parquet")["status"].item() == expected
    before = journal.read_bytes()
    assert recover_preboot_jobs(journal, generation_roots={"fixture": root}) == []
    assert journal.read_bytes() == before


def test_recovery_preserves_current_boot_jobs_and_rejects_ambiguous_time(tmp_path, monkeypatch):
    journal, root, old = interrupted_fixture(tmp_path, monkeypatch)
    live = dict(old, attempt=2, started_at_utc="2024-01-01T03:00:00+03:00")  # exactly boot
    done = dict(old, attempt=3, status="complete", exit_code=0)
    write_batches(journal, [pl.DataFrame([old, live, done], schema=JOURNAL_SCHEMA)], JOURNAL_SCHEMA)
    assert len(recover_preboot_jobs(journal)) == 1
    rows = pl.read_parquet(journal).to_dicts()
    assert rows[1] == live and rows[2] == done
    assert pl.read_parquet(root / "audit/generation_state.parquet")["status"].item() == "running"
    naive = dict(live, started_at_utc="2023-12-31T23:00:00")
    write_batches(journal, [pl.DataFrame([naive], schema=JOURNAL_SCHEMA)], JOURNAL_SCHEMA)
    before = journal.read_bytes()
    with pytest.raises(ValueError, match="timezone-aware"):
        recover_preboot_jobs(journal)
    assert journal.read_bytes() == before


def test_reboot_recovery_is_exclusive_and_new_attempt_keeps_history(tmp_path, monkeypatch):
    journal, root, _ = interrupted_fixture(tmp_path, monkeypatch)
    with FileLock(str(journal) + ".lock"):
        with pytest.raises(Timeout):
            recover_preboot_jobs(journal, generation_roots={"fixture": root})
    assert pl.read_parquet(journal)["status"].item() == "running"
    result = run_job([sys.executable, "-c", "pass"], cwd=tmp_path, journal=journal,
        dataset="fixture", stage="generator", model="all", config_hash="fixture", poll_seconds=.02)
    assert result["attempt"] == 2 and result["status"] == "complete"
    assert pl.read_parquet(journal)["status"].to_list() == ["interrupted", "complete"]
    assert recover_preboot_jobs(tmp_path / "absent.parquet") == []
    assert not (tmp_path / "absent.parquet").exists()
