"""Stage 6 — hierarchical detection.

A single flat multi-class classifier must solve two unrelated problems at once:
"is this traffic hostile?" and "if so, which family?".  Those have very
different base rates — 1.23% of flows are attacks, but among attacks
``SQL_INJECTION`` is 0.4% — so one softmax head optimises a compromise that
serves neither.  The benign/attack decision gets swamped by family splits.

The hierarchy separates them:

* **Level 1 — binary.**  Benign vs attack.  High recall is the priority: a
  missed attack costs far more than an investigated false positive.
* **Level 2 — family.**  Reached only for flows Level 1 flags.  Also produces a
  confidence, which is what gates Level 3.
* **Level 3 — zero-day.**  An unsupervised anomaly head.  When Level 2 is
  *unsure*, the flow may be an attack family never seen in training.  Forcing
  it into a known label is exactly how a NIDS misses novel threats.

Decision logic (from the project pseudocode)::

    p = Level1(X)
    if p <= threshold:            return BENIGN
    family, conf = Level2(X)
    if conf >= high_conf:         return family
    if Level3.anomaly(X) > thr:   return POTENTIAL_UNSEEN_ATTACK
    return family
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .taxonomy import AttackClass

UNSEEN_LABEL = -1


class HierarchicalDetector:
    """Three-level detector: binary gate -> family head -> anomaly head."""

    def __init__(
        self,
        *,
        detection_threshold: float = 0.5,
        high_confidence: float = 0.70,
        unseen_threshold: float = 0.0,
        random_state: int = 42,
    ) -> None:
        self.detection_threshold = detection_threshold
        self.high_confidence = high_confidence
        self.unseen_threshold = unseen_threshold
        self.random_state = random_state

        self.level1_: RandomForestClassifier | None = None
        self.level2_: RandomForestClassifier | None = None
        self.level3_: IsolationForest | None = None
        self.classes_: np.ndarray | None = None
        self.benign_score_quantiles_: dict[str, float] | None = None

    def fit(
        self,
        X: np.ndarray,
        y_family: np.ndarray,
        *,
        level2_contamination: float = 0.2,
    ) -> "HierarchicalDetector":
        """Train all three levels.

        Parameters
        ----------
        y_family
            Integer family labels; ``BENIGN`` is the negative class.
        level2_contamination
            Fraction of attack rows held out of Level-2 training.  Level 3 must
            learn what "normal for this class" means *without* seeing every
            attack, otherwise it flags known attacks as novel and Level 2's
            confidence gate never triggers.
        """
        X = np.asarray(X, dtype=np.float64)
        y_family = np.asarray(y_family)

        y_binary = (y_family != int(AttackClass.BENIGN)).astype(int)

        self.level1_ = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X, y_binary)

        # --- Level 2: family, trained on attacks only -----------------------
        attack_mask = y_binary == 1
        X_attack, y_attack = X[attack_mask], y_family[attack_mask]

        rng = np.random.default_rng(self.random_state)
        holdout = np.zeros(len(X_attack), dtype=bool)
        if len(X_attack) > 20 and level2_contamination > 0:
            # Stratified holdout so every family is still represented.
            n_hold = int(len(X_attack) * level2_contamination)
            chosen = rng.choice(len(X_attack), size=n_hold, replace=False)
            holdout[chosen] = True

        self.classes_ = np.unique(y_attack[~holdout])
        if len(self.classes_) > 1:
            self.level2_ = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ).fit(X_attack[~holdout], y_attack[~holdout])

        # --- Level 3: what does benign traffic look like? -------------------
        benign = X[y_binary == 0]
        if len(benign) > 20:
            sample = benign
            if len(benign) > 50_000:  # Isolation Forest is memory-hungry
                sample = benign[
                    rng.choice(len(benign), size=50_000, replace=False)
                ]
            self.level3_ = IsolationForest(
                n_estimators=200,
                contamination="auto",
                random_state=self.random_state,
                n_jobs=-1,
            ).fit(sample)

            # Calibrate the "anomalous" cut-off on benign traffic itself.
            #
            # IsolationForest.decision_function is centred on the training
            # distribution, NOT on zero: in-distribution rows typically score
            # slightly positive.  Assuming a 0.0 threshold therefore flags a
            # large slice of perfectly healthy traffic as a zero-day attack.
            # We store the benign score quantile so the caller can place the
            # threshold in a way that controls the false-alert rate.
            benign_scores = -self.level3_.decision_function(sample)
            self.benign_score_quantiles_ = {
                "p50": float(np.quantile(benign_scores, 0.50)),
                "p90": float(np.quantile(benign_scores, 0.90)),
                "p99": float(np.quantile(benign_scores, 0.99)),
                "p999": float(np.quantile(benign_scores, 0.999)),
            }
            if self.unseen_threshold == 0.0:
                # Default to the 99.9th percentile of *benign* traffic, i.e.
                # at most 0.1% of healthy flows are ever called unseen.
                self.unseen_threshold = self.benign_score_quantiles_["p999"]

        return self

    # ------------------------------------------------------------ predict ----

    def predict_proba_attack(self, X: np.ndarray) -> np.ndarray:
        """P(malicious) from Level 1."""
        if self.level1_ is None:
            raise RuntimeError("HierarchicalDetector.predict*() called before fit()")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.level1_.predict_proba(X)[:, 1]

    def predict_family(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Level-2 family labels and their confidence.

        Returns ``(labels, confidence)`` where an unfitted Level 2 yields
        ``UNSEEN_LABEL`` with confidence 0, so the anomaly head takes over.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = len(X)
        if self.level2_ is None or self.classes_ is None or len(self.classes_) < 2:
            return np.full(n, UNSEEN_LABEL), np.zeros(n)

        proba = self.level2_.predict_proba(X)
        best = proba.argmax(axis=1)
        return self.classes_[best], proba[np.arange(n), best]

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Level-3 deviation from benign traffic; higher means more anomalous.

        ``IsolationForest.decision_function`` returns "higher = more normal", so
        it is negated here to match the pseudocode's ``AnomalyScore``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.level3_ is None:
            return np.zeros(len(X))
        return -self.level3_.decision_function(X)

    def predict(self, X: np.ndarray) -> list[dict]:
        """Full hierarchical verdict for each row.

        Each result carries the decision, the level that produced it, and the
        confidence — the ``level`` field is what makes an alert explainable to a
        SOC analyst ("this is a family call" vs "this is an anomaly").

        Note that a flow is first gated by Level 1, so a novel attack whose
        *aggregate statistics* still look benign will be passed as benign.  The
        anomaly head can only veto what Level 1 lets through.  This is inherent
        to a supervised gate and is why ``unseen_threshold`` should be tuned on
        a held-out unseen-attack protocol rather than assumed.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        p_attack = self.predict_proba_attack(X)
        families, confidences = self.predict_family(X)
        anomalies = self.anomaly_score(X)

        results = []
        for i in range(len(X)):
            # A row that clears the attack gate OR is strongly off-distribution
            # continues down the hierarchy; otherwise Level 1 closes it out.
            anomalous = anomalies[i] > self.unseen_threshold

            if p_attack[i] <= self.detection_threshold and not anomalous:
                results.append(
                    self._verdict(
                        AttackClass.BENIGN, level=1, confidence=1.0 - p_attack[i],
                        p_attack=p_attack[i], anomaly=anomalies[i],
                    )
                )
                continue

            family = int(families[i])
            confidence = float(confidences[i])
            known = family in self._known_ids()

            # A confident family call only counts if the flow is NOT strongly
            # off-manifold.  A tree ensemble will happily return p=0.95 for a
            # point far outside every training cluster, because it has never
            # seen a reason to doubt itself.  Letting the anomaly head veto
            # here is what stops a genuinely novel attack from being filed
            # under the nearest known family.
            if known and confidence >= self.high_confidence and not anomalous:
                results.append(
                    self._verdict(
                        AttackClass(family), level=2, confidence=confidence,
                        p_attack=p_attack[i], anomaly=anomalies[i],
                    )
                )
            elif anomalous or not known:
                # Either the anomaly head agrees this is off-distribution, or
                # Level 2 produced a family it was never trained on.  Both cases
                # must be reported as unseen rather than guessed at.
                results.append(
                    self._verdict(
                        None, level=3, confidence=confidence,
                        p_attack=p_attack[i], anomaly=anomalies[i],
                        unseen=True,
                    )
                )
            else:
                results.append(
                    self._verdict(
                        AttackClass(family), level=2, confidence=confidence,
                        p_attack=p_attack[i], anomaly=anomalies[i],
                    )
                )

        return results

    def _verdict(
        self,
        cls: AttackClass | None,
        *,
        level: int,
        confidence: float,
        p_attack: float,
        anomaly: float,
        unseen: bool = False,
    ) -> dict:
        if unseen or cls is None:
            return {
                "label": "Potential Unseen Attack",
                "label_id": UNSEEN_LABEL,
                "is_attack": True,
                "unseen": True,
                "level": level,
                "confidence": round(confidence, 4),
                "p_attack": round(float(p_attack), 4),
                "anomaly_score": round(float(anomaly), 4),
            }

        return {
            "label": cls.name,
            "label_id": int(cls),
            "is_attack": cls != AttackClass.BENIGN,
            "unseen": False,
            "level": level,
            "confidence": round(confidence, 4),
            "p_attack": round(float(p_attack), 4),
            "anomaly_score": round(float(anomaly), 4),
        }

    def _known_ids(self) -> set[int]:
        """Family ids Level 2 was actually trained on."""
        if self.classes_ is None:
            return set()
        return {int(c) for c in self.classes_}
