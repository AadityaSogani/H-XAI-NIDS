"""Canonical attack-class taxonomy for H-XAI-NIDS.

The raw CICIDS-2017 parquet files use human-readable labels such as
``"Web Attack ΓÇô Brute Force"``.  That middle character is an *en dash* (U+2013)
which is stored correctly in the parquet but renders as mojibake when printed
through a ``cp1252`` Windows console (``ΓÇö``/``∩┐╜``).  It also varies in
capitalisation and whitespace between files.

To avoid a whole class of subtle bugs, every label is normalised *once* here
into a stable, ASCII-only slug, and the rest of the pipeline only ever deals
with :class:`AttackClass` members.
"""
from __future__ import annotations

import unicodedata
from enum import IntEnum


def _normalise(label: str) -> str:
    """Lower-case, strip accents/punctuation and collapse whitespace."""
    # NFKD splits accented chars into base + combining marks; we keep ASCII only.
    text = unicodedata.normalize("NFKD", str(label))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Collapse every run of non-alphanumeric characters into a single space.
    cleaned = []
    for ch in text:
        cleaned.append(ch if ch.isalnum() else " ")
    return " ".join("".join(cleaned).split())


class AttackClass(IntEnum):
    """Canonical taxonomy used across training, serving and the web UI.

    The integer value doubles as the Level-2 (family) classifier target, so the
    ordering is fixed and must not be reshuffled without retraining.
    """

    BENIGN = 0
    BOT = 1
    PORT_SCAN = 2
    BRUTE_FORCE = 3
    XSS = 4
    SQL_INJECTION = 5


# Raw-label lookup built from normalised strings, so encoding quirks, casing
# differences and stray punctuation are all tolerated.
_LOOKUP = {
    "benign": AttackClass.BENIGN,
    "bot": AttackClass.BOT,
    "botnet": AttackClass.BOT,
    "portscan": AttackClass.PORT_SCAN,
    "port scan": AttackClass.PORT_SCAN,
    "web attack brute force": AttackClass.BRUTE_FORCE,
    "brute force": AttackClass.BRUTE_FORCE,
    "web attack xss": AttackClass.XSS,
    "xss": AttackClass.XSS,
    "web attack sql injection": AttackClass.SQL_INJECTION,
    "sql injection": AttackClass.SQL_INJECTION,
    "infiltration": AttackClass.BENIGN,  # placeholder; no such rows in our subset
}

# Pretty display names for the UI / reports.
DISPLAY_NAMES: dict[AttackClass, str] = {
    AttackClass.BENIGN: "Benign",
    AttackClass.BOT: "Bot",
    AttackClass.PORT_SCAN: "Port Scan",
    AttackClass.BRUTE_FORCE: "Brute Force",
    AttackClass.XSS: "Cross-Site Scripting",
    AttackClass.SQL_INJECTION: "SQL Injection",
}

# Colour tokens used by the dashboard (kept in one place so UI + reports agree).
SEVERITY_COLORS: dict[AttackClass, str] = {
    AttackClass.BENIGN: "#22c55e",
    AttackClass.BOT: "#f97316",
    AttackClass.PORT_SCAN: "#eab308",
    AttackClass.BRUTE_FORCE: "#f43f5e",
    AttackClass.XSS: "#a855f7",
    AttackClass.SQL_INJECTION: "#ec4899",
}


def to_class(label: str) -> AttackClass:
    """Map an arbitrary raw label to its canonical :class:`AttackClass`.

    Raises
    ------
    KeyError
        If the label is not recognised.  Failing loudly is deliberate: a silent
        fallback would hide genuinely new attack families.
    """
    key = _normalise(label)
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        raise KeyError(
            f"Unrecognised label {label!r} (normalised to {key!r}). "
            "Add it to _LOOKUP in taxonomy.py if it is a legitimate new class."
        ) from exc


def is_attack(cls: AttackClass) -> bool:
    """Level-1 target: True for anything that is not benign."""
    return cls != AttackClass.BENIGN
