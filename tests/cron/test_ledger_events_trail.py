"""Append-only per-run event trail and the frozen outcome / delivery vocabulary.

The trail is the run's own history (claimed → started → finished/failed → delivered or
not_delivered, or reclaimed), not a projection of mutable job-store fields, and it is never
rewritten after the fact.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


def _kinds(events):
    return [event["kind"] for event in events]


def test_successful_run_records_a_full_trail(executions):
    record = executions.create_execution("trail-job", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(
        record["id"], success=True, delivery_outcome="delivered",
        delivery_target="telegram:42",
    )

    events = executions.list_execution_events(execution_id=record["id"])
    assert _kinds(events) == ["claimed", "started", "finished", "delivered"]
    assert all(event["job_id"] == "trail-job" for event in events)


def test_failed_run_records_failed_and_not_delivered(executions):
    record = executions.create_execution("fail-job", source="direct")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=False, error="provider exploded")

    events = executions.list_execution_events(execution_id=record["id"])
    assert _kinds(events) == ["claimed", "started", "failed", "not_delivered"]
    assert events[-1]["detail"] == executions.DELIVERY_NONE


def test_reclaimed_attempt_appends_a_reclaimed_event(executions, monkeypatch):
    record = executions.create_execution("reclaim-trail", source="direct")
    executions.mark_execution_running(record["id"])
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)

    assert executions.recover_interrupted_executions() == 1

    events = executions.list_execution_events(execution_id=record["id"])
    assert "reclaimed" in _kinds(events)
    assert events[-1]["detail"], "a reclaim must record the real reason"


def test_events_are_append_only_and_bound_to_the_job(executions):
    record = executions.create_execution("bound-job", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True, delivery_outcome="delivered")
    before = executions.list_execution_events(execution_id=record["id"])

    # A terminal attempt cannot be rewritten, so its trail cannot grow or change.
    assert executions.finish_execution(record["id"], success=False, error="late") is None
    after = executions.list_execution_events(execution_id=record["id"])
    assert after == before
    assert executions.list_execution_events(job_id="bound-job") == before


def test_delivered_run_records_its_real_destination(executions):
    record = executions.create_execution("delivered-job", source="builtin")
    executions.mark_execution_running(record["id"])
    finished = executions.finish_execution(
        record["id"], success=True, delivery_outcome="delivered",
        delivery_target="whatsapp:+971500000000",
    )

    assert finished["delivery_outcome"] == "delivered"
    assert finished["delivery_target"] == "whatsapp:+971500000000"
    delivered = [
        event for event in executions.list_execution_events(execution_id=record["id"])
        if event["kind"] == "delivered"
    ]
    assert delivered and delivered[-1]["target"] == "whatsapp:+971500000000"


def test_missing_destination_is_recorded_as_none_not_as_delivered(executions):
    record = executions.create_execution("no-target", source="builtin")
    finished = executions.finish_execution(record["id"], success=True)

    assert finished["delivery_outcome"] == executions.DELIVERY_NONE
    assert finished["delivery_target"] == executions.DELIVERY_NONE
    assert executions.normalize_delivery_outcome(None) == "none"
    assert executions.normalize_delivery_outcome("") == "none"
    assert executions.normalize_delivery_outcome("queued") == "queued"


def test_bound_destination_survives_the_terminal_write(executions):
    """The scheduler binds the lane first; finish_execution must not overwrite it with 'none'."""
    record = executions.create_execution("bound-target", source="builtin")
    executions.mark_execution_running(record["id"])
    assert executions.set_delivery_target(record["id"], "telegram:-100555") is True

    finished = executions.finish_execution(
        record["id"], success=True, delivery_outcome="delivered")

    assert finished["delivery_target"] == "telegram:-100555"
    delivered = [
        event for event in executions.list_execution_events(execution_id=record["id"])
        if event["kind"] == "delivered"
    ]
    assert delivered and delivered[-1]["target"] == "telegram:-100555"


def test_outcome_vocabulary_is_frozen(executions):
    assert executions.EXECUTION_OUTCOMES == ("completed", "failed", "unknown")
    completed = executions.create_execution("done", source="builtin")
    executions.finish_execution(completed["id"], success=True)
    assert executions.execution_outcome(executions.get_execution(completed["id"])) == "completed"
    assert executions.execution_outcome(None) == "unknown"

    inflight = executions.create_execution("inflight", source="builtin")
    # An attempt with no finish record cannot prove an outcome: unknown, not completed.
    assert executions.execution_outcome(executions.get_execution(inflight["id"])) == "unknown"
