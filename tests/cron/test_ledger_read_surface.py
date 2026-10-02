"""The pure read surface the EVE snapshot service consumes.

Every field is derived from the live store/ledger; a field the ledger cannot prove is ``None``.
The surface never synthesises a default (no assumed completion, no recomputed next-run), and it
never writes.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME (job store) plus an isolated execution ledger."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return {"home": tmp_path, "executions": executions_mod}


def _create_job(schedule="every 5m", name="snap"):
    from cron.jobs import create_job

    return create_job(prompt="x", schedule=schedule, name=name)


def test_job_with_no_ledger_row_reports_no_last_run(env):
    from cron.ledger_snapshot import job_ledger_snapshot

    job = _create_job(name="never-ran")
    snapshot = job_ledger_snapshot(job["id"])

    assert snapshot is not None
    assert snapshot["job_id"] == job["id"]
    assert snapshot["name"] == "never-ran"
    # No ledger row ⇒ no last run. Never a synthesised completed run.
    assert snapshot["last_run"] is None


def test_unknown_schedule_reports_null_next_run(env):
    from cron.jobs import load_jobs, save_jobs
    from cron.ledger_snapshot import job_ledger_snapshot

    job = _create_job(schedule="every 5m", name="weird")
    records = load_jobs()
    record = next(r for r in records if r["id"] == job["id"])
    # A stored stamp the surface must NOT trust once the schedule is not one Hermes honours.
    record["schedule"] = {"kind": "wat", "display": "whenever"}
    record["next_run_at"] = "2030-01-01T00:00:00+00:00"
    save_jobs(records)

    snapshot = job_ledger_snapshot(job["id"])
    assert snapshot["schedule"] is None
    assert snapshot["next_run_at"] is None, "an unknown schedule cannot prove a next run"


def test_known_schedule_reports_the_stored_next_run(env):
    from cron.ledger_snapshot import job_ledger_snapshot

    job = _create_job(schedule="every 5m", name="regular")
    snapshot = job_ledger_snapshot(job["id"])

    assert snapshot["schedule_display"]
    assert snapshot["next_run_at"] == job["next_run_at"] or (
        isinstance(snapshot["next_run_at"], str)
    )
    from datetime import datetime

    datetime.fromisoformat(snapshot["next_run_at"])  # parses ⇒ a real stored instant


def test_snapshot_reports_the_last_run_and_its_outcome(env):
    from cron.jobs import create_job
    from cron.ledger_snapshot import job_ledger_snapshot

    executions = env["executions"]
    job = create_job(prompt="x", schedule="every 5m", name="ran")
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(
        record["id"], success=True, delivery_outcome="delivered",
        delivery_target="telegram:99",
    )

    snapshot = job_ledger_snapshot(job["id"])
    last = snapshot["last_run"]
    assert last["id"] == record["id"]
    assert last["outcome"] == "completed"
    assert last["delivery_outcome"] == "delivered"
    assert last["delivery_target"] == "telegram:99"
    assert [event["kind"] for event in last["events"]] == [
        "claimed", "started", "finished", "delivered",
    ]


def test_snapshot_for_unknown_job_is_none(env):
    from cron.ledger_snapshot import job_ledger_snapshot

    assert job_ledger_snapshot("does-not-exist") is None


def test_snapshot_is_a_pure_read(env):
    from cron.jobs import create_job
    from cron.ledger_snapshot import job_ledger_snapshot

    executions = env["executions"]
    job = create_job(prompt="x", schedule="every 5m", name="read-only")
    record = executions.create_execution(job["id"], source="builtin")
    executions.finish_execution(record["id"], success=True, delivery_outcome="delivered")

    before_rows = executions.list_executions(job_id=job["id"])
    before_events = executions.list_execution_events(execution_id=record["id"])
    first = job_ledger_snapshot()
    second = job_ledger_snapshot()
    assert first == second
    assert executions.list_executions(job_id=job["id"]) == before_rows
    assert executions.list_execution_events(execution_id=record["id"]) == before_events


def test_snapshot_lists_store_jobs_and_ledger_only_ids(env):
    from cron.jobs import create_job
    from cron.ledger_snapshot import job_ledger_snapshot

    executions = env["executions"]
    job = create_job(prompt="x", schedule="every 5m", name="listed")
    executions.create_execution("orphan-job", source="external")

    entries = {entry["job_id"]: entry for entry in job_ledger_snapshot()}
    assert job["id"] in entries
    assert entries[job["id"]]["name"] == "listed"
    # A ledger-only job has no store record, so store fields stay None but the run is visible.
    assert entries["orphan-job"]["name"] is None
    assert entries["orphan-job"]["last_run"]["id"]
