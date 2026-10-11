"""Unit tests for the stratified splitter."""
from __future__ import annotations

import numpy as np
import pytest

from ml.splitting import RANDOM_STATE, Split, stratified_split
from ml.taxonomy import AttackClass


def make_imbalanced_labels() -> np.ndarray:
    """Mirror the real corpus: 445,844 benign rows plus five attack families.

    SQL_INJECTION gets exactly 21 rows, as in the real dataset — this is the
    class the splitter must protect.
    """
    families = [
        (AttackClass.BENIGN, 445_844),
        (AttackClass.BOT, 1_437),
        (AttackClass.PORT_SCAN, 1_956),
        (AttackClass.BRUTE_FORCE, 1_470),
        (AttackClass.XSS, 652),
        (AttackClass.SQL_INJECTION, 21),
    ]

    y = np.concatenate(
        [np.full(count, int(cls), dtype=np.int64) for cls, count in families]
    )

    rng = np.random.default_rng(0)
    rng.shuffle(y)

    assert y.size == 451_380
    return y


def test_tiers_are_disjoint_and_exhaustive() -> None:
    y = make_imbalanced_labels()
    split = stratified_split(y)

    split.assert_disjoint()
    assert split.train.size + split.validation.size + split.test.size == y.size


def test_rare_class_stays_entirely_in_train() -> None:
    """21 SQL-injection rows must not be sliced into tiny val/test slivers."""
    y = make_imbalanced_labels()
    split = stratified_split(y)

    rare = np.flatnonzero(y == int(AttackClass.SQL_INJECTION))
    assert rare.size == 21

    assert set(rare.tolist()) <= set(split.train.tolist())
    assert not (set(rare.tolist()) & set(split.validation.tolist()))
    assert not (set(rare.tolist()) & set(split.test.tolist()))


def test_min_per_class_two_would_stratify_21_rows() -> None:
    """Documents why the default is 3, not 2.

    With ``min_per_class=2`` the threshold is 20, so 21 SQL-injection rows *are*
    stratified — leaving only ~2 validation and ~4 test rows.  Precision/recall
    on a 4-row test slice is noise, so the default deliberately protects them.
    """
    y = make_imbalanced_labels()
    split = stratified_split(y, min_per_class=2)

    sqli = int(AttackClass.SQL_INJECTION)
    assert (y[split.validation] == sqli).sum() > 0
    assert (y[split.test] == sqli).sum() > 0
    # ...but the slice is far too small to measure anything meaningful.
    assert (y[split.test] == sqli).sum() < 10


def test_common_classes_appear_in_every_tier() -> None:
    y = make_imbalanced_labels()
    split = stratified_split(y)

    for cls in (AttackClass.BENIGN, AttackClass.BOT, AttackClass.PORT_SCAN):
        c = int(cls)
        assert (y[split.train] == c).sum() > 0, f"{cls.name} missing from train"
        assert (y[split.validation] == c).sum() > 0, f"{cls.name} missing from validation"
        assert (y[split.test] == c).sum() > 0, f"{cls.name} missing from test"


def test_attack_rate_is_preserved_across_tiers() -> None:
    y = make_imbalanced_labels()
    split = stratified_split(y)
    overall = (y != int(AttackClass.BENIGN)).mean()

    for name, idx in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        rate = (y[idx] != int(AttackClass.BENIGN)).mean()
        assert rate == pytest.approx(overall, abs=0.01), f"{name} attack rate drifted"


def test_split_is_deterministic() -> None:
    y = make_imbalanced_labels()
    a = stratified_split(y, random_state=RANDOM_STATE)
    b = stratified_split(y, random_state=RANDOM_STATE)
    assert np.array_equal(a.train, b.train)
    assert np.array_equal(a.test, b.test)
    assert np.array_equal(a.validation, b.validation)


def test_split_sizes_are_within_tolerance() -> None:
    y = make_imbalanced_labels()
    split = stratified_split(y)
    n = y.size

    # SQL_INJECTION (21 rows) is forced into train, so val/test land slightly
    # under the nominal fractions.
    assert split.test.size == pytest.approx(0.2 * n, rel=0.05)
    assert split.validation.size == pytest.approx(0.1 * n, rel=0.15)


def test_single_class_is_degenerate() -> None:
    y = np.zeros(50, dtype=np.int64)
    split = stratified_split(y)

    assert split.train.size == 50
    assert split.validation.size == 0
    assert split.test.size == 0


def test_all_rare_classes_degenerate() -> None:
    y = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
    split = stratified_split(y)

    assert split.train.size == 6
    assert split.validation.size == 0
    assert split.test.size == 0


def test_summary_reports_sizes() -> None:
    y = make_imbalanced_labels()
    summary = stratified_split(y).summary()

    assert set(summary) == {"train", "validation", "test"}
    assert sum(summary.values()) == y.size


def test_indices_are_sorted() -> None:
    y = make_imbalanced_labels()
    split = stratified_split(y)

    for idx in (split.train, split.validation, split.test):
        assert np.all(np.diff(idx) > 0)


def test_assert_disjoint_detects_overlap() -> None:
    bad = Split(
        train=np.array([0, 1, 2]),
        validation=np.array([2, 3]),
        test=np.array([4]),
    )
    with pytest.raises(AssertionError):
        bad.assert_disjoint()
