"""Unit tests for the explainability layer."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.explain import SurrogateExplainer, TreeShapExplainer

FEATURES = ["f_a", "f_b", "f_c", "f_d"]


@pytest.fixture
def fitted_rf():
    """A tiny forest where f_a dominates the decision."""
    rng = np.random.default_rng(17)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
    return model, X, y


# --------------------------------------------------------------- TreeSHAP ----


def test_global_importance_covers_every_feature(fitted_rf):
    model, X, _ = fitted_rf
    ranking = TreeShapExplainer(model, FEATURES, background=X).global_importance()

    assert len(ranking) == len(FEATURES)
    assert {r["feature"] for r in ranking} == set(FEATURES)


def test_global_importance_is_ranked_descending(fitted_rf):
    model, X, _ = fitted_rf
    ranking = TreeShapExplainer(model, FEATURES, background=X).global_importance()

    values = [r["mean_abs_shap"] for r in ranking]
    assert values == sorted(values, reverse=True)
    assert [r["rank"] for r in ranking] == list(range(1, len(FEATURES) + 1))


def test_global_importance_identifies_the_dominant_feature(fitted_rf):
    """f_a generated the label, so it must top the ranking."""
    model, X, _ = fitted_rf
    ranking = TreeShapExplainer(model, FEATURES, background=X).global_importance()

    assert ranking[0]["feature"] == "f_a"


def test_local_explanation_returns_top_k(fitted_rf):
    model, X, _ = fitted_rf
    explainer = TreeShapExplainer(model, FEATURES, background=X)
    result = explainer.local_explanation(X[0], top_k=3)

    assert len(result) == 3
    assert {a["feature"] for a in result} <= set(FEATURES)


def test_local_explanation_ranks_by_magnitude(fitted_rf):
    model, X, _ = fitted_rf
    explainer = TreeShapExplainer(model, FEATURES, background=X)
    result = explainer.local_explanation(X[0], top_k=4)

    magnitudes = [abs(a["contribution"]) for a in result]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_local_explanation_carries_the_row_value(fitted_rf):
    model, X, _ = fitted_rf
    explainer = TreeShapExplainer(model, FEATURES, background=X)
    row = X[0]

    result = explainer.local_explanation(row, top_k=4)

    for attr in result:
        idx = FEATURES.index(attr["feature"])
        # Attributions are rounded to 4 dp for the API payload.
        assert attr["value"] == pytest.approx(float(row[idx]), abs=1e-4)


def test_local_explanation_accepts_a_1d_row(fitted_rf):
    model, X, _ = fitted_rf
    explainer = TreeShapExplainer(model, FEATURES, background=X)

    assert len(explainer.local_explanation(X[3], top_k=2)) == 2


def test_global_importance_subsamples_large_backgrounds(fitted_rf):
    """A 10k background with sample_size=500 must still return all features."""
    model, X, _ = fitted_rf
    rng = np.random.default_rng(0)
    big = rng.normal(size=(10_000, 4))

    ranking = TreeShapExplainer(
        model, FEATURES, background=big
    ).global_importance(sample_size=500)

    assert len(ranking) == len(FEATURES)



# --------------------------------------------------------------- surrogate ----


def test_surrogate_reports_fidelity(fitted_rf):
    model, X, _ = fitted_rf
    result = SurrogateExplainer(
        lambda Z: model.predict_proba(Z)[:, 1], FEATURES
    ).explain(X[0], X)

    assert "fidelity_r2" in result
    assert -1.0 <= result["fidelity_r2"] <= 1.0
    assert "intercept" in result


def test_surrogate_scores_high_on_a_locally_linear_model():
    """A linear model must be reproduced almost exactly by a Ridge surrogate."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    weights = np.array([2.0, -1.0, 0.5, 0.0])

    result = SurrogateExplainer(lambda Z: Z @ weights, FEATURES).explain(X[0], X)

    assert result["fidelity_r2"] > 0.98
    assert result["contributions"][0]["feature"] == "f_a"


def test_surrogate_covers_every_feature(fitted_rf):
    model, X, _ = fitted_rf
    result = SurrogateExplainer(
        lambda Z: model.predict_proba(Z)[:, 1], FEATURES
    ).explain(X[0], X)

    assert {a["feature"] for a in result["contributions"]} == set(FEATURES)


def test_surrogate_perturbations_stay_between_real_flows(fitted_rf):
    """Samples must lie on segments between the row and real neighbours.

    This is the property LIME-style independent jitter violates: it invents
    physically impossible flows, so the surrogate ends up explaining a region
    the model is never asked about.
    """
    model, X, _ = fitted_rf
    explainer = SurrogateExplainer(
        lambda Z: model.predict_proba(Z)[:, 1], FEATURES, n_samples=200
    )

    rng = np.random.default_rng(explainer.random_state)
    row = X[0].reshape(1, -1)
    neighbours = explainer._nearest(row, X)
    noise = explainer._perturb(row, neighbours, rng)

    assert (noise >= np.minimum(row, neighbours.min(axis=0)) - 1e-9).all()
    assert (noise <= np.maximum(row, neighbours.max(axis=0)) + 1e-9).all()


def test_surrogate_handles_an_empty_background(fitted_rf):
    """With no neighbours to draw on, it must still return an explanation."""
    model, X, _ = fitted_rf
    explainer = SurrogateExplainer(
        lambda Z: model.predict_proba(Z)[:, 1], FEATURES
    )

    result = explainer.explain(X[0], np.empty((0, 4)))

    assert len(result["contributions"]) == len(FEATURES)


def test_surrogate_is_deterministic(fitted_rf):
    model, X, _ = fitted_rf
    explainer = SurrogateExplainer(
        lambda Z: model.predict_proba(Z)[:, 1], FEATURES
    )

    a = explainer.explain(X[0], X)
    b = explainer.explain(X[0], X)

    assert a["fidelity_r2"] == pytest.approx(b["fidelity_r2"])
    assert [c["contribution"] for c in a["contributions"]] == pytest.approx(
        [c["contribution"] for c in b["contributions"]]
    )
