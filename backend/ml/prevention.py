"""Intrusion prevention — turning a detection verdict into a countermeasure.

Detection and prevention answer two different questions.  Detection (the
hierarchy in ``hierarchy.py``) answers *whether* a flow is hostile and *which
family* it belongs to.  Prevention answers the operational follow-up: *what do
we do about it?*  This module is that second half, and nothing else — it never
re-classifies traffic, it only reacts to the verdict it is handed.

Design constraints, mirroring the detector's own philosophy:

* **Confidence-gated enforcement.**  A confident attack enforces the family's
  countermeasure; an uncertain one only raises an ``ALERT``.  A system that
  drops traffic on a shaky call does more harm than the attack, so disruption
  is reserved for calls the detector is sure of — the same ``high_confidence``
  line that gates the family head.
* **Repeat-offender escalation.**  A source already on the blocklist has its
  severity raised and its TTL extended: a persistent attacker should not be
  treated like a one-shot probe.
* **Honest enforcement boundary.**  Enforcement here is *simulated*: the engine
  maintains a live, TTL-based blocklist and emits a ready-to-apply packet-filter
  rule, but it does not touch the host network stack.  Wiring :func:`render_rule`
  output into a real firewall is an integration concern left to the deployment.

The blocklist is keyed by whatever the caller can attribute the verdict to — a
source IP is ideal, a destination port is the fallback, and with no context at
all the decision is still computed (and surfaced) but nothing is keyed.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    "ActionType",
    "Policy",
    "RESPONSE_POLICY",
    "ACTION_CONFIDENCE_THRESHOLD",
    "ENFORCING_ACTIONS",
    "Blocklist",
    "PreventionEngine",
    "blocklist_key",
    "render_rule",
    "decide",
]

# The confidence a detection verdict must carry before its countermeasure is
# enforced.  Kept identical to the detector's Level-2 ``high_confidence`` so the
# two layers agree on what "sure" means.
ACTION_CONFIDENCE_THRESHOLD = 0.70


class ActionType(str, Enum):
    """The countermeasure a prevention decision prescribes."""

    ALLOW = "ALLOW"            # benign — let it through
    ALERT = "ALERT"            # log/raise, but do not disrupt
    RATE_LIMIT = "RATE_LIMIT"  # throttle a noisy/probing source
    DROP = "DROP"              # discard the flow (application-layer attacks)
    BLOCK = "BLOCK"            # blackhole the source
    QUARANTINE = "QUARANTINE"  # isolate an unknown-but-anomalous source


# Actions that actually put something on the blocklist / firewall.  ALLOW and
# ALERT are passive: they inform, they never disrupt.
ENFORCING_ACTIONS = frozenset(
    {
        ActionType.RATE_LIMIT,
        ActionType.DROP,
        ActionType.BLOCK,
        ActionType.QUARANTINE,
    }
)


@dataclass(frozen=True, slots=True)
class Policy:
    """Per-family response: the action, how long it lasts, and its severity."""

    action: ActionType
    ttl_seconds: int
    severity: int  # 0 (info) .. 5 (critical); matches the UI severity ramp
    note: str


# The response table.  Severity reuses the frontend's 0..5 ramp so a family is
# the same colour everywhere.  TTLs are a judgement call: web-app attacks are
# per-request so short, credential attacks persist longer, and an unseen attack
# is quarantined for triage rather than silently blocked.
RESPONSE_POLICY: dict[str, Policy] = {
    "BENIGN": Policy(ActionType.ALLOW, 0, 0, "no countermeasure required"),
    "PORT_SCAN": Policy(
        ActionType.RATE_LIMIT, 300, 2, "reconnaissance — throttle the source"
    ),
    "BOT": Policy(
        ActionType.BLOCK, 3600, 3, "suspected botnet — blackhole the source"
    ),
    "BRUTE_FORCE": Policy(
        ActionType.BLOCK, 1800, 4, "credential attack — block the source"
    ),
    "XSS": Policy(
        ActionType.DROP, 600, 4, "cross-site scripting — drop the request"
    ),
    "SQL_INJECTION": Policy(
        ActionType.DROP, 600, 5, "SQL injection — drop the request"
    ),
    "Potential Unseen Attack": Policy(
        ActionType.QUARANTINE,
        900,
        5,
        "off-manifold traffic — quarantine for triage",
    ),
}

# Fallback for any label the table does not know: treat it as an unseen attack
# (quarantine), never as benign.  Failing safe is the whole point.
_UNKNOWN_POLICY = RESPONSE_POLICY["Potential Unseen Attack"]


@dataclass(slots=True)
class BlockEntry:
    """One active entry on the blocklist."""

    key: str
    action: str
    severity: int
    reason: str
    created_at: float
    expires_at: float
    hits: int = 1


class Blocklist:
    """In-memory, TTL-based blocklist of enforced prevention subjects.

    Entries expire on their own so a temporary rate-limit does not become a
    permanent ban, and a re-hit refreshes the TTL and bumps ``hits`` so the
    escalation in :func:`decide` has signal to work with.  ``now`` is injectable
    purely so tests can drive time without sleeping.
    """

    def __init__(self, now: Callable[[], float] = time.time) -> None:
        self._entries: dict[str, BlockEntry] = {}
        self._now = now

    def add(
        self,
        key: str,
        *,
        action: str,
        severity: int,
        reason: str,
        ttl_seconds: int,
    ) -> BlockEntry:
        """Add or refresh an entry.  A live entry is extended, not restarted."""
        now = self._now()
        existing = self._entries.get(key)
        if existing is not None and existing.expires_at > now:
            existing.action = action
            existing.severity = max(existing.severity, severity)
            existing.expires_at = now + ttl_seconds
            existing.hits += 1
            existing.reason = reason
            return existing

        entry = BlockEntry(
            key=key,
            action=action,
            severity=severity,
            reason=reason,
            created_at=now,
            expires_at=now + ttl_seconds,
        )
        self._entries[key] = entry
        return entry

    def is_blocked(self, key: str | None) -> bool:
        if key is None:
            return False
        entry = self._entries.get(key)
        return entry is not None and entry.expires_at > self._now()

    def remove(self, key: str) -> bool:
        return self._entries.pop(key, None) is not None

    def purge_expired(self) -> int:
        now = self._now()
        expired = [k for k, e in self._entries.items() if e.expires_at <= now]
        for k in expired:
            del self._entries[k]
        return len(expired)

    def clear(self) -> int:
        n = len(self._entries)
        self._entries.clear()
        return n

    def entries(self) -> list[dict[str, Any]]:
        """Active entries as serialisable dicts, soonest-expiry first."""
        self.purge_expired()
        now = self._now()
        ordered = sorted(self._entries.values(), key=lambda e: e.expires_at)
        return [
            {
                "key": e.key,
                "action": e.action,
                "severity": e.severity,
                "reason": e.reason,
                "hits": e.hits,
                "created_at": e.created_at,
                "expires_at": e.expires_at,
                "ttl_remaining": max(0.0, e.expires_at - now),
            }
            for e in ordered
        ]


def blocklist_key(context: Mapping[str, Any] | None) -> str | None:
    """Best available subject to key a prevention action on.

    A source IP is what you actually want to block; a destination port is a
    coarse fallback; with neither there is nothing sensible to key on, so the
    decision is still computed but left unattributed.
    """
    if not context:
        return None
    src = context.get("src_ip")
    if src:
        return f"ip:{src}"
    port = context.get("dst_port")
    if port not in (None, "", 0):
        return f"port:{port}"
    return None


def render_rule(
    action: str, key: str | None, context: Mapping[str, Any] | None
) -> str:
    """A representative packet-filter rule for the chosen action.

    Advisory by design: this is the command an operator *would* run to enforce
    the decision on a real host.  The engine does not execute it.
    """
    enforcing = {a.value for a in ENFORCING_ACTIONS}
    if action not in enforcing:
        return f"# {action}: passive — no packet-filter rule"

    src_ip = (context or {}).get("src_ip")
    port = (context or {}).get("dst_port")

    if action == ActionType.RATE_LIMIT.value:
        if src_ip:
            return (
                f"iptables -A INPUT -s {src_ip} -m hashlimit "
                f"--hashlimit-upto 10/min --hashlimit-mode srcip -j ACCEPT "
                f"&& iptables -A INPUT -s {src_ip} -j DROP"
            )
        return f"# rate-limit requires a source IP; key={key}"

    # DROP / BLOCK / QUARANTINE all blackhole; they differ in intent, not rule.
    if src_ip:
        return f"iptables -A INPUT -s {src_ip} -j DROP"
    if port not in (None, "", 0):
        return f"iptables -A INPUT -p tcp --dport {port} -j DROP"
    return f"# {action}: no IP/port to build a rule from; key={key}"


def decide(
    verdict: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    blocklist: Blocklist | None = None,
    *,
    mode: str = "prevent",
    action_confidence_threshold: float = ACTION_CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """Map a detection verdict onto a prevention action.

    Parameters
    ----------
    verdict
        A hierarchical verdict dict (``label``, ``is_attack``, ``confidence``…).
    context
        Optional, model-irrelevant metadata (src_ip, dst_port…) used only to key
        the action.  Never fed back into the detector.
    blocklist
        The live blocklist, used for repeat-offender escalation and enforcement.
    mode
        ``"prevent"`` enforces; ``"monitor"`` is a dry run (``enforced=False``).
    """
    label = str(verdict.get("label", ""))
    confidence = float(verdict.get("confidence", 0.0))
    p_attack = float(verdict.get("p_attack", 0.0))
    is_attack = bool(verdict.get("is_attack", False))

    policy = RESPONSE_POLICY.get(label, _UNKNOWN_POLICY)
    key = blocklist_key(context)

    # --- benign: nothing to do --------------------------------------------
    if not is_attack or label == "BENIGN":
        return _decision(
            action=ActionType.ALLOW,
            policy=policy,
            key=key,
            context=context,
            rationale=f"Classified as {label or 'benign'}; no countermeasure required.",
        )

    # --- attack, but below the confidence bar: alert, do not disrupt ------
    if confidence < action_confidence_threshold:
        return _decision(
            action=ActionType.ALERT,
            policy=policy,
            key=key,
            context=context,
            gated=True,
            ttl_seconds=0,
            rationale=(
                f"{label} flagged (P(attack)={p_attack:.3f}) but family "
                f"confidence {confidence:.2f} < {action_confidence_threshold:.2f}; "
                f"alerting only — traffic not disrupted."
            ),
        )

    # --- confident attack: enforce the family countermeasure --------------
    escalated = blocklist is not None and blocklist.is_blocked(key)
    severity = policy.severity
    ttl = policy.ttl_seconds
    if escalated:
        severity = min(5, severity + 1)
        ttl *= 2

    enforced = False
    if (
        mode == "prevent"
        and blocklist is not None
        and key is not None
        and policy.action in ENFORCING_ACTIONS
    ):
        blocklist.add(
            key,
            action=policy.action.value,
            severity=severity,
            reason=f"{label} · confidence {confidence:.2f}",
            ttl_seconds=ttl,
        )
        enforced = True

    return _decision(
        action=policy.action,
        policy=policy,
        key=key,
        context=context,
        enforced=enforced,
        escalated=escalated,
        severity=severity,
        ttl_seconds=ttl,
        rationale=_enforce_rationale(
            label, confidence, policy, escalated, ttl, enforced, key
        ),
    )


def _enforce_rationale(label, confidence, policy, escalated, ttl, enforced, key):
    base = (
        f"{label} detected (confidence {confidence:.2f}) → "
        f"{policy.action.value} for {ttl}s."
    )
    if escalated:
        base += " Repeat offender: severity raised, TTL extended."
    if not enforced:
        if key is None:
            base += " No IP/port context, so nothing to key — decision advisory only."
        else:
            base += f" Monitor mode: {key} not actually added."
    return base


def _decision(
    *,
    action: ActionType,
    policy: Policy,
    key: str | None,
    context: Mapping[str, Any] | None,
    rationale: str,
    enforced: bool = False,
    gated: bool = False,
    escalated: bool = False,
    severity: int | None = None,
    ttl_seconds: int | None = None,
) -> dict[str, Any]:
    return {
        "action": action.value,
        "enforced": enforced,
        "gated": gated,
        "escalated": escalated,
        "severity": policy.severity if severity is None else severity,
        "ttl_seconds": policy.ttl_seconds if ttl_seconds is None else ttl_seconds,
        "rationale": rationale,
        "rule": render_rule(action.value, key, context),
        "subject": key,
    }


class PreventionEngine:
    """Stateful wrapper: the response policy plus its live blocklist.

    A single instance is the app's source of truth for prevention state, the
    same way ``ModelStore`` is for the model.  :meth:`decide` is the only entry
    point the routes need.
    """

    def __init__(
        self,
        blocklist: Blocklist | None = None,
        *,
        action_confidence_threshold: float = ACTION_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.blocklist = blocklist or Blocklist()
        self.action_confidence_threshold = action_confidence_threshold

    def decide(
        self,
        verdict: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        *,
        mode: str = "prevent",
    ) -> dict[str, Any]:
        return decide(
            verdict,
            context,
            self.blocklist,
            mode=mode,
            action_confidence_threshold=self.action_confidence_threshold,
        )

    def unblock(self, key: str) -> bool:
        return self.blocklist.remove(key)



