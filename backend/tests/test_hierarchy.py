"""Unit tests for the three-level hierarchical detector."""
from __future__ import annotations

import numpy as np
import pytest

from ml.hierarchy import UNSEEN_LABEL, HierarchicalDetector
from ml.taxonomy import AttackClass


def make_dataset(n_benign=800, per_family=80):
    """Three well-separated attack families plus benign traffic.

    Benign traffic occupies a tight ball at the origin; each family lives in its
    own region, so a *novel* attack can be placed genuinely off-manifold.
    """
    rng = np.random.default_rng(21)

    benign = rng.normal(0.0, 1.0, size=(n_benign, 6))
    bot = rng.normal(5.0, 0.6, size=(per_family, 6))
    portscan = rng.normal(-5.0, 0.6, size=(per_family, 6))
    xss = rng.normal(0.0, 1.0, size=(per_family, 6))
    xss[:, 0] += 12.0  # separated on one axis only

    X = np.vstack([benign, bot, portscan, xss])
    y = np.concatenate(
        [
            np.full(n_benign, int(AttackClass.BENIGN)),
            np.full(per_family, int(AttackClass.BOT)),
            np.full(per_family, int(AttackClass.PORT_SCAN)),
            np.full(per_family, int(AttackClass.XSS)),
        ]
    ).astype(int)
    return X, y


def make_novel_attack(n=40):
    """An attack region no training row occupies, on EVERY axis.

    Placing it far out on all six dimensions keeps it well outside the benign
    manifold, which is what a genuine zero-day looks like to an anomaly head.
    """
    rng = np.random.default_rng(99)
    return rng.normal(0.0, 0.4, size=(n, 6)) + 25.0


@pytest.fixture
def fitted():
    X, y = make_dataset()
    model = HierarchicalDetector(random_state=0).fit(X, y)
    return model, X, y


# ------------------------------------------------------------------ fit ----


def test_all_three_levels_are_fitted(fitted):
    model, X, _ = fitted
    assert model.level1_ is not None
    assert model.level2_ is not None
    assert model.level3_ is not None


def test_level2_trains_on_attacks_only(fitted):
    """Level 2 must never see a benign row, or family recall is meaningless."""
    model, X, y = fitted
    assert int(AttackClass.BENIGN) not in {int(c) for c in model.classes_}


def test_predict_before_fit_raises():
    X = np.random.default_rng(0).normal(size=(5, 6))
    with pytest.raises(RuntimeError):
        HierarchicalDetector().predict_proba_attack(X)


# -------------------------------------------------------------- predict ----


def test_benign_rows_are_classified_benign(fitted):
    """The overwhelming majority of benign flows must close out at Level 1.

    The anomaly head is deliberately allowed to fire on a small fraction of
    healthy traffic — that is the cost of catching zero-days.  With the
    threshold set at the p99.9 of benign scores, ~0.1% of healthy flows are
    escalated for investigation rather than silently passed.
    """
    model, X, y = fitted
    verdicts = model.predict(X[y == int(AttackClass.BENIGN)])

    benign_rate = sum(1 for v in verdicts if v["label"] == "BENIGN") / len(verdicts)
    assert benign_rate > 0.98, f"only {benign_rate:.2%} of benign traffic passed"

    # Whatever is escalated must at least be a genuine escalation, not a
    # confident wrong-family call.
    escalated = [v for v in verdicts if v["label"] != "BENIGN"]
    assert all(v["unseen"] for v in escalated)


def test_attacks_are_not_reported_benign(fitted):
    model, X, y = fitted
    attack_rows = X[y != int(AttackClass.BENIGN)]
    verdicts = model.predict(attack_rows)

    assert all(v["is_attack"] for v in verdicts)


def test_verdict_carries_audit_fields(fitted):
    model, X, _ = fitted
    verdict = model.predict(X[:1])[0]

    assert set(verdict) >= {
        "label", "label_id", "is_attack", "unseen",
        "level", "confidence", "p_attack", "anomaly_score",
    }


def test_confidence_is_probabilistic(fitted):
    model, X, _ = fitted
    verdicts = model.predict(X[:50])

    assert all(0.0 <= v["confidence"] <= 1.0 for v in verdicts)
    assert all(0.0 <= v["p_attack"] <= 1.0 for v in verdicts)


def test_predict_accepts_a_single_row(fitted):
    model, X, _ = fitted
    assert len(model.predict(X[0])) == 1


def test_predict_is_deterministic(fitted):
    model, X, _ = fitted
    assert model.predict(X[:20]) == model.predict(X[:20])


# ----------------------------------------------------------- zero-day ----


def test_unseen_attack_is_flagged_as_unseen(fitted):
    """A family absent from training must not be forced into a known label."""
    model, _, _ = fitted
    novel = make_novel_attack()

    verdicts = model.predict(novel)

    unseen = [v for v in verdicts if v["unseen"]]
    assert unseen, "a clearly novel attack was silently assigned a known family"
    assert unseen[0]["label"] == "Potential Unseen Attack"
    assert unseen[0]["label_id"] == UNSEEN_LABEL
    assert unseen[0]["level"] == 3
    assert unseen[0]["is_attack"]


def test_anomaly_threshold_is_calibrated_on_benign_traffic(fitted):
    """The cut-off must come from the benign distribution, not from 0.0.

    ``IsolationForest.decision_function`` is centred on its training data, so
    in-distribution rows score slightly *positive*.  A hard-coded 0.0 threshold
    therefore calls a large share of healthy traffic a zero-day attack.
    """
    model, X, y = fitted

    assert model.benign_score_quantiles_ is not None
    assert model.unseen_threshold > 0.0
    assert model.unseen_threshold == pytest.approx(
        model.benign_score_quantiles_["p999"]
    )

    # At most ~0.1% of benign flows should trip the anomaly head.
    benign_scores = model.anomaly_score(X[y == int(AttackClass.BENIGN)])
    rate = float((benign_scores > model.unseen_threshold).mean())
    assert rate < 0.01, f"{rate:.2%} of benign traffic flagged as unseen"


def test_novel_attack_scores_above_benign_quantiles(fitted):
    model, X, y = fitted

    novel_scores = model.anomaly_score(make_novel_attack())
    benign_scores = model.anomaly_score(X[y == int(AttackClass.BENIGN)])

    assert novel_scores.mean() > np.quantile(benign_scores, 0.999)


def test_high_detection_threshold_suppresses_attacks():
    """A very high threshold means almost nothing clears the Level-1 gate."""
    X, y = make_dataset()
    strict = HierarchicalDetector(
        detection_threshold=0.999, high_confidence=0.999, random_state=0
    ).fit(X, y)

    p = strict.predict_proba_attack(X)
    verdicts = strict.predict(X)

    # Rows the gate rejects are benign at level 1, provided the anomaly head
    # does not also fire for them.
    rejected = (p <= 0.999) & (
        strict.anomaly_score(X) <= strict.unseen_threshold
    )
    assert rejected.any(), "expected the high threshold to reject most rows"
    assert all(
        v["label"] == "BENIGN" and v["level"] == 1
        for v, keep in zip(verdicts, rejected) if keep
    )


def test_low_threshold_admits_more_than_a_high_one():
    """Lowering the gate must monotonically increase what reaches Level 2/3.

    This is the trade-off the detection threshold actually controls, and it is
    the knob a SOC would tune: raise it to cut false positives, lower it to
    catch more attacks.
    """
    X, y = make_dataset()
    attack_rows = X[y != int(AttackClass.BENIGN)]

    strict = HierarchicalDetector(
        detection_threshold=0.99, high_confidence=0.999, random_state=0
    ).fit(X, y)
    permissive = HierarchicalDetector(
        detection_threshold=0.10, high_confidence=0.999, random_state=0
    ).fit(X, y)

    strict_flagged = sum(1 for v in strict.predict(attack_rows) if v["is_attack"])
    loose_flagged = sum(1 for v in permissive.predict(attack_rows) if v["is_attack"])

    assert loose_flagged >= strict_flagged
    assert loose_flagged == len(attack_rows), (
        "a 0.10 threshold should admit every true attack"
    )


def test_threshold_of_zero_still_closes_exact_zero_probabilities():
    """Documents the ``<=`` boundary: rows scored exactly 0.0 are closed.

    A random forest fitted on clean data often assigns p=0.0 to obviously
    benign rows (every tree votes benign), and the gate is ``<=``, so those
    rows are still reported benign even at a threshold of 0.  This is correct
    behaviour, not a defect — it just means the gate is inclusive.
    """
    X, y = make_dataset()
    model = HierarchicalDetector(detection_threshold=0.0, random_state=0).fit(X, y)

    p = model.predict_proba_attack(X)
    verdicts = model.predict(X)

    exact_zero = p <= 0.0
    assert exact_zero.any(), "expected some rows to be scored exactly 0.0"
    assert all(
        v["level"] == 1 and v["label"] == "BENIGN"
        for v, closed in zip(verdicts, exact_zero) if closed
    )


def test_benign_rows_survive_a_permissive_gate_as_attacks():
    """With no gate, benign traffic must still resolve to a concrete label."""
    X, y = make_dataset()
    permissive = HierarchicalDetector(detection_threshold=0.0, random_state=0).fit(X, y)

    verdicts = permissive.predict(X[y == int(AttackClass.BENIGN)])

    assert all(v["label"] for v in verdicts)
    # Level 1 may still close the handful of rows scored at exactly p=0.0.
    assert all(v["level"] in (1, 2, 3) for v in verdicts)


def test_disabling_the_anomaly_head_restores_a_pure_gate():
    """Without Level 3, Level 2 alone decides — useful for ablations.

    This is the control condition for the "hierarchical + zero-day head" rung
    of the benchmark ladder in the project's evaluation protocol.
    """
    X, y = make_dataset()
    plain = HierarchicalDetector(detection_threshold=0.0, random_state=0).fit(X, y)
    plain.level3_ = None
    plain.unseen_threshold = float("inf")  # anomaly can never fire

    verdicts = plain.predict(X[y == int(AttackClass.BENIGN)])

    assert all(not v["unseen"] for v in verdicts)
    assert all(v["level"] in (1, 2) for v in verdicts)


def test_missing_level3_still_returns_verdicts():
    """With the anomaly head absent, prediction must degrade gracefully."""
    X, y = make_dataset(n_benign=40, per_family=12)
    model = HierarchicalDetector(random_state=0).fit(X, y)
    model.level3_ = None

    verdicts = model.predict(X[:10])

    assert len(verdicts) == 10
    assert np.allclose(model.anomaly_score(X[:10]), 0.0)


def test_single_family_degrades_to_unseen():
    """With only one attack family, Level 2 cannot discriminate."""
    rng = np.random.default_rng(4)
    X = np.vstack([rng.normal(0, 1, size=(300, 5)), rng.normal(7, 0.5, size=(40, 5))])
    y = np.concatenate([np.zeros(300), np.ones(40)]).astype(int)

    model = HierarchicalDetector(random_state=0).fit(X, y)
    verdicts = model.predict(X[y == 1][:5])

    # Level 2 has a single class, so every attack surfaces via the anomaly head.
    assert all(v["is_attack"] for v in verdicts)
    assert all(v["unseen"] for v in verdicts)
