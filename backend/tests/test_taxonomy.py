"""Unit tests for the canonical label taxonomy."""
from __future__ import annotations

import pytest

from ml.taxonomy import (
    AttackClass,
    DISPLAY_NAMES,
    is_attack,
    to_class,
)

# The real labels shipped in the parquet files, including the en-dash (U+2013)
# that renders as mojibake in a cp1252 console.
REAL_LABELS = {
    "Benign": AttackClass.BENIGN,
    "Bot": AttackClass.BOT,
    "PortScan": AttackClass.PORT_SCAN,
    "Web Attack \u2013 Brute Force": AttackClass.BRUTE_FORCE,
    "Web Attack \u2013 XSS": AttackClass.XSS,
    "Web Attack \u2013 Sql Injection": AttackClass.SQL_INJECTION,
}


@pytest.mark.parametrize(("raw", "expected"), list(REAL_LABELS.items()))
def test_real_labels_resolve(raw: str, expected: AttackClass) -> None:
    assert to_class(raw) is expected


@pytest.mark.parametrize(
    "raw",
    ["benign", "BENIGN", "  benign  ", "Benign\t"],
)
def test_case_and_whitespace_insensitive(raw: str) -> None:
    assert to_class(raw) is AttackClass.BENIGN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("port scan", AttackClass.PORT_SCAN),
        ("Port-Scan", AttackClass.PORT_SCAN),
        ("Web-Attack: Brute Force", AttackClass.BRUTE_FORCE),
        ("web attack sql injection", AttackClass.SQL_INJECTION),
        ("botnet", AttackClass.BOT),
    ],
)
def test_punctuation_variants(raw: str, expected: AttackClass) -> None:
    assert to_class(raw) is expected


def test_unknown_label_raises() -> None:
    """Unknown families must fail loudly, not silently fall back."""
    with pytest.raises(KeyError):
        to_class("DDoS Hulk")


def test_class_integers_are_stable() -> None:
    """The int values are the Level-2 target encoding; they must not drift."""
    assert int(AttackClass.BENIGN) == 0
    assert int(AttackClass.SQL_INJECTION) == 5
    assert len(AttackClass) == 6


def test_display_names_cover_every_class() -> None:
    for cls in AttackClass:
        assert DISPLAY_NAMES[cls]


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (AttackClass.BENIGN, False),
        (AttackClass.BOT, True),
        (AttackClass.PORT_SCAN, True),
        (AttackClass.BRUTE_FORCE, True),
        (AttackClass.XSS, True),
        (AttackClass.SQL_INJECTION, True),
    ],
)
def test_is_attack(cls: AttackClass, expected: bool) -> None:
    assert is_attack(cls) is expected
