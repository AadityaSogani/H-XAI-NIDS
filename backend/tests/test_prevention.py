"""Unit and integration tests for the intrusion-prevention (IPS) layer.

The policy engine is pure, so most tests drive :func:`ml.prevention.decide`
directly with hand-built verdicts.  A final block exercises the real
``/prevent`` and ``/prevent/blocklist`` endpoints through TestClient, and is
skipped when no trained artifact is present (the same guard the smoke test uses).
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ml.prevention import (
    ACTION_CONFIDENCE_THRESHOLD,
    ENFORCING_ACTIONS,
    ActionType,
    Blocklist,
    PreventionEngine,
    blocklist_key,
    decide,
    render_rule,
)


def verdict(label, *, is_attack, confidence, p_attack=0.95, unseen=False, level=2):
    """Build a minimal detection verdict dict, mirroring the hierarchy's shape."""
    return {
        "label": label,
        "label_id": -1 if unseen else 0,
        "is_attack": is_attack,
        "unseen": unseen,
        "level": level,
        "confidence": confidence,
        "p_attack": p_attack,
        "anomaly_score": 0.1,
    }


# --------------------------------------------------------------- policy ----


@pytest.mark.parametrize(
    "label,expected",
    [
        ("BENIGN", ActionType.ALLOW),
        ("PORT_SCAN", ActionType.RATE_LIMIT),
        ("BOT", ActionType.BLOCK),
        ("BRUTE_FORCE", ActionType.BLOCK),
        ("XSS", ActionType.DROP),
        ("SQL_INJECTION", ActionType.DROP),
        ("Potential Unseen Attack", ActionType.QUARANTINE),
    ],
)
def test_each_family_maps_to_its_policy_action(label, expected):
    """A confident attack must resolve to exactly the action its family dictates."""
    d = decide(verdict(label, is_attack=label != "BENIGN", confidence=0.95))
    assert d["action"] == expected.value


def test_benign_is_allowed_and_never_enforced():
    d = decide(verdict("BENIGN", is_attack=False, confidence=1.0))
    assert d["action"] == "ALLOW"
    assert not d["enforced"]
    assert not d["gated"]


def test_unknown_label_fails_safe_to_quarantine():
    """An unrecognised family must never be allowed through as benign."""
    d = decide(verdict("SOME_NEW_ATTACK", is_attack=True, confidence=0.99))
    assert d["action"] == "QUARANTINE"


def test_enforcing_actions_are_exactly_the_disruptive_ones():
    assert {a.value for a in ENFORCING_ACTIONS} == {
        "RATE_LIMIT",
        "DROP",
        "BLOCK",
        "QUARANTINE",
    }


# ---------------------------------------------------- confidence gate ----


def test_low_confidence_attack_only_alerts():
    """Below the gate, an attack must be surfaced but NOT disrupt traffic."""
    d = decide(
        verdict("XSS", is_attack=True, confidence=0.40),
        {"src_ip": "1.2.3.4"},
    )
    assert d["action"] == "ALERT"
    assert d["gated"] is True
    assert d["enforced"] is False


def test_gate_boundary_is_inclusive():
    """Exactly at the threshold counts as confident (uses ``<`` not ``<=``)."""
    at = decide(
        verdict("XSS", is_attack=True, confidence=ACTION_CONFIDENCE_THRESHOLD),
        {"src_ip": "1.2.3.4"},
    )
    below = decide(
        verdict("XSS", is_attack=True, confidence=ACTION_CONFIDENCE_THRESHOLD - 1e-6),
        {"src_ip": "1.2.3.4"},
    )
    assert at["action"] == "DROP"
    assert below["action"] == "ALERT"


# --------------------------------------------------------- enforcement ----


def test_prevent_mode_enforces_into_blocklist():
    bl = Blocklist()
    d = decide(
        verdict("BRUTE_FORCE", is_attack=True, confidence=0.9),
        {"src_ip": "9.9.9.9"},
        bl,
        mode="prevent",
    )
    assert d["enforced"] is True
    assert bl.is_blocked("ip:9.9.9.9")


def test_monitor_mode_never_enforces():
    bl = Blocklist()
    d = decide(
        verdict("BRUTE_FORCE", is_attack=True, confidence=0.9),
        {"src_ip": "9.9.9.9"},
        bl,
        mode="monitor",
    )
    assert d["action"] == "BLOCK"  # the decision is still reported
    assert d["enforced"] is False
    assert not bl.is_blocked("ip:9.9.9.9")
    assert "Monitor mode" in d["rationale"]


def test_no_context_means_no_key_and_advisory_only():
    bl = Blocklist()
    d = decide(
        verdict("BOT", is_attack=True, confidence=0.9),
        None,
        bl,
        mode="prevent",
    )
    assert d["subject"] is None
    assert d["enforced"] is False
    assert not bl.entries()


def test_repeat_offender_escalates_severity_and_ttl():
    bl = Blocklist()
    v = verdict("BOT", is_attack=True, confidence=0.95)
    first = decide(v, {"src_ip": "9.9.9.9"}, bl, mode="prevent")
    second = decide(v, {"src_ip": "9.9.9.9"}, bl, mode="prevent")
    assert first["escalated"] is False
    assert second["escalated"] is True
    assert second["severity"] == first["severity"] + 1
    assert second["ttl_seconds"] == first["ttl_seconds"] * 2
    assert bl.entries()[0]["hits"] == 2


def test_severity_escalation_is_capped_at_five():
    bl = Blocklist()
    # SQL_INJECTION is already severity 5; a repeat must not overflow to 6.
    v = verdict("SQL_INJECTION", is_attack=True, confidence=0.99)
    decide(v, {"src_ip": "9.9.9.9"}, bl, mode="prevent")
    second = decide(v, {"src_ip": "9.9.9.9"}, bl, mode="prevent")
    assert second["severity"] == 5


# ---------------------------------------------------------- blocklist ----


def test_blocklist_entry_expires_by_ttl():
    """A block must lapse on its own; a rate-limit is not a permanent ban."""
    clock = {"t": 1000.0}
    bl = Blocklist(now=lambda: clock["t"])
    bl.add("ip:1.1.1.1", action="BLOCK", severity=4, reason="test", ttl_seconds=60)

    assert bl.is_blocked("ip:1.1.1.1")
    clock["t"] += 59
    assert bl.is_blocked("ip:1.1.1.1")
    clock["t"] += 2  # now past the 60s TTL
    assert not bl.is_blocked("ip:1.1.1.1")
    assert bl.entries() == []


def test_rehit_refreshes_ttl_without_resetting_hits():
    clock = {"t": 1000.0}
    bl = Blocklist(now=lambda: clock["t"])
    bl.add("ip:1.1.1.1", action="BLOCK", severity=4, reason="r", ttl_seconds=60)
    clock["t"] += 30
    bl.add("ip:1.1.1.1", action="BLOCK", severity=4, reason="r", ttl_seconds=60)
    entry = bl.entries()[0]
    assert entry["hits"] == 2
    assert entry["expires_at"] == 1030.0 + 60  # extended from the re-hit, not the start


def test_blocklist_remove_and_clear():
    bl = Blocklist()
    bl.add("ip:1.1.1.1", action="BLOCK", severity=4, reason="r", ttl_seconds=60)
    assert bl.remove("ip:1.1.1.1") is True
    assert bl.remove("ip:1.1.1.1") is False
    bl.add("ip:2.2.2.2", action="BLOCK", severity=4, reason="r", ttl_seconds=60)
    assert bl.clear() == 1
    assert bl.entries() == []


# ------------------------------------------------------- key & rules ----


def test_blocklist_key_prefers_source_ip():
    assert blocklist_key({"src_ip": "1.2.3.4", "dst_port": 80}) == "ip:1.2.3.4"


def test_blocklist_key_falls_back_to_port():
    assert blocklist_key({"dst_port": 80}) == "port:80"


def test_blocklist_key_is_none_without_context():
    assert blocklist_key(None) is None
    assert blocklist_key({}) is None


def test_render_rule_for_block_and_drop_uses_source_ip():
    assert render_rule("BLOCK", "ip:1.2.3.4", {"src_ip": "1.2.3.4"}) == (
        "iptables -A INPUT -s 1.2.3.4 -j DROP"
    )
    assert render_rule("DROP", "ip:1.2.3.4", {"src_ip": "1.2.3.4"}) == (
        "iptables -A INPUT -s 1.2.3.4 -j DROP"
    )


def test_render_rule_rate_limit_mentions_hashlimit():
    rule = render_rule("RATE_LIMIT", "ip:1.2.3.4", {"src_ip": "1.2.3.4"})
    assert "hashlimit" in rule and "1.2.3.4" in rule


def test_render_rule_passive_actions_have_no_rule():
    for action in ("ALLOW", "ALERT"):
        assert "passive" in render_rule(action, None, None)


def test_render_rule_port_fallback_without_ip():
    rule = render_rule("BLOCK", "port:22", {"dst_port": 22})
    assert "--dport 22" in rule


# ---------------------------------------------------- engine wrapper ----


def test_engine_shares_one_blocklist_across_calls():
    """State must persist between calls, or escalation and listing break."""
    engine = PreventionEngine()
    v = verdict("BOT", is_attack=True, confidence=0.95)
    engine.decide(v, {"src_ip": "5.5.5.5"}, mode="prevent")
    d = engine.decide(v, {"src_ip": "5.5.5.5"}, mode="prevent")
    assert d["escalated"] is True
    assert engine.unblock("ip:5.5.5.5") is True


# ------------------------------------------- HTTP integration (opt-in) ----


@pytest.fixture
def client():
    from app.config import settings as app_settings

    if not (app_settings.artifact_path / "manifest.json").exists():
        pytest.skip("no trained artifact; run `python -m ml.train`")
    from app.main import app

    with TestClient(app) as c:
        yield c


def _attack_row(c):
    """A real flow row from the trained background, so detection is realistic."""
    from app.config import settings as app_settings
    from ml.registry import NIDSArtifact

    artifact = NIDSArtifact.load(app_settings.artifact_path)
    background = np.load(app_settings.artifact_path / "background.npy")
    return {
        name: float(background[0][i]) for i, name in enumerate(artifact.feature_names)
    }


def test_prevent_endpoint_returns_verdict_and_action(client):
    r = client.post(
        "/prevent",
        json={
            "features": _attack_row(client),
            "context": {"src_ip": "203.0.113.7"},
            "mode": "prevent",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "verdict" in body and "prevention" in body
    assert body["prevention"]["action"] in {
        "ALLOW",
        "ALERT",
        "RATE_LIMIT",
        "DROP",
        "BLOCK",
        "QUARANTINE",
    }


def test_prevent_monitor_mode_does_not_populate_blocklist(client):
    r = client.post(
        "/prevent",
        json={
            "features": _attack_row(client),
            "context": {"src_ip": "198.51.100.9"},
            "mode": "monitor",
        },
    )
    assert r.status_code == 200, r.text
    # Whatever the verdict, monitor mode never enforces.
    assert r.json()["prevention"]["enforced"] is False


def test_prevent_rejects_bad_mode(client):
    r = client.post(
        "/prevent",
        json={
            "features": _attack_row(client),
            "mode": "aggressive",  # not in (prevent|monitor)
        },
    )
    assert r.status_code == 422


def test_prevent_missing_features_is_400(client):
    r = client.post("/prevent", json={"features": {"Total Fwd Packets": 1}})
    assert r.status_code == 400


def test_blocklist_endpoint_shape(client):
    r = client.get("/prevent/blocklist")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == len(body["entries"])


def test_unblock_unknown_key_is_404(client):
    r = client.delete("/prevent/blocklist/ip:0.0.0.0")
    assert r.status_code == 404

