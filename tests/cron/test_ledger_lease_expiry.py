"""Explicit execution/fire leases: an expired lease is never reported as running.

Covers the increment that gives an attempt an explicit ``lease_until`` (derived from the shared
``FIRE_CLAIM_TTL_SECONDS``) instead of re-inferring liveness from the claim timestamp, and that
records the real reclaim reason when the lease lapses.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


@pytest.fixture()
def executions(monkeypatch, tmp_path):
    import cron.executions as executions_mod

    monkeypatch.setattr(
        executions_mod, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db"
    )
    return executions_mod


@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_claim_carries_an_explicit_lease_derived_from_the_shared_ttl(temp_home):
    from cron.constants import FIRE_CLAIM_TTL_SECONDS
    from cron.jobs import claim_job_for_fire, claim_lease_until, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="lease")
    assert claim_job_for_fire(job["id"]) is True

    claim = get_job(job["id"])["fire_claim"]
    claimed_at = datetime.fromisoformat(claim["at"])
    lease_until = claim_lease_until(claim)
    assert lease_until is not None, "the claim must carry an explicit lease, not infer one"
    assert lease_until == claimed_at + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS)


def test_expired_claim_is_not_reported_live_even_with_a_live_owner(
    temp_home, monkeypatch
):
    """Past its explicit lease a claim is stale, whatever the owner pid says."""
    from cron.constants import FIRE_CLAIM_TTL_SECONDS

    import cron.jobs as jobs

    job = jobs.create_job(prompt="x", schedule="every 5m", name="expiring")
    assert jobs.claim_job_for_fire(job["id"]) is True
    claim = jobs.get_job(job["id"])["fire_claim"]
    claimed_at = datetime.fromisoformat(claim["at"])

    # Owner pid is this very process (alive); only the lease has lapsed.
    monkeypatch.setattr(
        jobs, "_hermes_now",
        lambda: claimed_at + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS + 5),
    )
    assert jobs._claim_is_live(claim, jobs._hermes_now(), FIRE_CLAIM_TTL_SECONDS) is False
    # …and the fire slot is reclaimable without waiting for a dead pid.
    assert jobs.claim_job_for_fire(job["id"]) is True


def test_expired_lease_is_not_reported_as_running(executions, monkeypatch):
    record = executions.create_execution("lease-job", source="builtin")
    running = executions.mark_execution_running(record["id"])
    assert running["status"] == "running"
    assert running["lease_until"], "the attempt must carry an explicit lease"

    deadline = datetime.fromisoformat(running["lease_until"])
    monkeypatch.setattr(executions, "_hermes_now", lambda: deadline + timedelta(seconds=1))

    assert executions.execution_lease_expired(running) is True
    # The ownership read must NOT report a lapsed attempt as running, even though the owning
    # pid (this process) is still alive.
    assert executions.live_inflight_execution("lease-job") is None


def test_renewed_lease_keeps_a_live_run_reportable(executions):
    record = executions.create_execution("renewed", source="builtin")
    executions.mark_execution_running(record["id"])
    before = executions.get_execution(record["id"])["lease_until"]

    assert executions.renew_execution_lease(record["id"]) is True
    after = executions.get_execution(record["id"])["lease_until"]
    assert datetime.fromisoformat(after) > datetime.fromisoformat(before)
    assert executions.live_inflight_execution("renewed") is not None


def test_interrupted_expired_attempt_records_the_real_reason(executions, monkeypatch):
    """A lapsed lease is reclaimed as an expired lease — status unknown, reason explicit."""
    record = executions.create_execution("expired-lease", source="direct")
    executions.mark_execution_running(record["id"])
    deadline = datetime.fromisoformat(executions.get_execution(record["id"])["lease_until"])

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    monkeypatch.setattr(executions, "_hermes_now", lambda: deadline + timedelta(seconds=1))

    assert executions.recover_interrupted_executions() == 1
    recovered = executions.get_execution(record["id"])
    assert recovered["status"] == "unknown"
    assert recovered["error"] == executions._LEASE_EXPIRED_REASON
    assert "lease expired" in recovered["error"].lower()
    assert executions.execution_outcome(recovered) == "unknown"

    reclaimed = [
        event for event in executions.list_execution_events(execution_id=record["id"])
        if event["kind"] == "reclaimed"
    ]
    assert reclaimed and reclaimed[-1]["detail"] == executions._LEASE_EXPIRED_REASON


def test_interrupted_attempt_with_a_fresh_lease_keeps_the_owner_gone_reason(
    executions, monkeypatch
):
    """A dead owner inside its lease is not mislabelled as an expired lease."""
    record = executions.create_execution("dead-owner", source="direct")
    executions.mark_execution_running(record["id"])

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)

    assert executions.execution_lease_expired(executions.get_execution(record["id"])) is False
    assert executions.recover_interrupted_executions() == 1
    recovered = executions.get_execution(record["id"])
    assert recovered["status"] == "unknown"
    assert recovered["error"] == executions._OWNER_GONE_REASON


def test_fire_claim_heartbeat_renews_the_execution_lease(executions, temp_home, monkeypatch):
    """The scheduler's heartbeat hook keeps the execution lease in lockstep with the fire claim.

    Without it a run that legitimately outlives FIRE_CLAIM_TTL_SECONDS would read as an expired
    lease and stop being reported as running.
    """
    import cron.scheduler as scheduler_mod
    from cron.jobs import claim_job_for_fire, create_job

    job = create_job(prompt="x", schedule="every 5m", name="heartbeat-lease")
    assert claim_job_for_fire(job["id"]) is True
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])
    issued = datetime.fromisoformat(executions.get_execution(record["id"])["lease_until"])

    later = issued + timedelta(minutes=5)
    monkeypatch.setattr(executions, "_hermes_now", lambda: later)
    scheduler_mod._renew_execution_lease_best_effort(
        {"id": job["id"], "execution_id": record["id"]}, job["id"]
    )

    renewed = executions.get_execution(record["id"])
    assert datetime.fromisoformat(renewed["lease_until"]) > later
    assert executions.execution_lease_expired(renewed) is False
    assert executions.live_inflight_execution(job["id"]) is not None


def test_legacy_row_without_a_lease_uses_the_shared_ttl(executions, monkeypatch):
    """A pre-lease row reads its lease from FIRE_CLAIM_TTL_SECONDS over claimed_at."""
    from cron.constants import FIRE_CLAIM_TTL_SECONDS

    record = executions.create_execution("legacy", source="builtin")
    with executions._transaction() as conn:
        conn.execute("UPDATE executions SET lease_until=NULL WHERE id=?", (record["id"],))

    row = executions.get_execution(record["id"])
    claimed_at = datetime.fromisoformat(row["claimed_at"])
    monkeypatch.setattr(
        executions, "_hermes_now",
        lambda: claimed_at + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS - 1),
    )
    assert executions.execution_lease_expired(executions.get_execution(record["id"])) is False

    monkeypatch.setattr(
        executions, "_hermes_now",
        lambda: claimed_at + timedelta(seconds=FIRE_CLAIM_TTL_SECONDS + 1),
    )
    assert executions.execution_lease_expired(executions.get_execution(record["id"])) is True
    assert executions.live_inflight_execution("legacy") is None
