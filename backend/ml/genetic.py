"""Stage 4b — Genetic Algorithm wrapper for feature selection.

Univariate filters score features *one at a time*, so they are blind to
interactions: two individually weak features can jointly carry a great deal of
signal.  A wrapper searches subsets directly under a model's own objective.

**Fitness** (matching the project pseudocode)::

    fitness = 0.85 * PR_AUC - 0.15 * (n_selected / n_total)

The parsimony term is what stops the GA from simply keeping every feature —
without it the fitness is monotonic in subset size and the search is pointless.
PR-AUC is used rather than ROC-AUC because at a 1.23% attack rate ROC-AUC is
dominated by the benign majority and looks good even for useless classifiers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

PR_AUC_WEIGHT = 0.85
PARSIMONY_WEIGHT = 0.15


@dataclass(slots=True)
class GAResult:
    """Best subset found, plus enough history to plot convergence."""

    mask: np.ndarray
    selected: list[str]
    fitness: float
    pr_auc: float
    n_selected: int
    generations: int
    history: list[float]

    def summary(self) -> dict:
        return {
            "n_selected": self.n_selected,
            "fitness": self.fitness,
            "pr_auc": self.pr_auc,
            "generations": self.generations,
            "selected": self.selected,
        }


def _make_estimator(random_state: int) -> RandomForestClassifier:
    """Small, balanced RF used as the fitness proxy."""
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=8,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def _cross_val_pr_auc(
    subset: np.ndarray,
    y: np.ndarray,
    *,
    folds: int,
    random_state: int,
) -> float:
    """Out-of-fold PR-AUC, or 0.0 if the subset cannot be evaluated."""
    usable_folds = min(folds, int(np.bincount(y).min()))
    if usable_folds < 2:
        return 0.0

    splitter = StratifiedKFold(
        n_splits=usable_folds, shuffle=True, random_state=random_state
    )
    proba = cross_val_predict(
        _make_estimator(random_state),
        subset,
        y,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    return float(average_precision_score(y, proba))


class GeneticFeatureSelector:
    """Bit-string GA over a shortlist of candidate features.

    A population of boolean masks evolves under tournament selection, uniform
    crossover and bit-flip mutation, with elitism so the best subset is never
    lost between generations.
    """

    def __init__(
        self,
        *,
        population_size: int = 30,
        generations: int = 8,
        tournament_size: int = 3,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        cv_folds: int = 3,
        min_features: int = 4,
        random_state: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cv_folds = cv_folds
        self.min_features = min_features
        self.random_state = random_state

    def fitness(self, mask: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate one subset.  Degenerate/empty subsets score 0."""
        n_selected = int(mask.sum())
        if n_selected < self.min_features:
            return 0.0

        try:
            pr_auc = _cross_val_pr_auc(
                X[:, mask.astype(bool)], y,
                folds=self.cv_folds, random_state=self.random_state,
            )
        except Exception:
            # A singular or unfittable subset is treated as useless.
            return 0.0

        parsimony = n_selected / mask.size
        return PR_AUC_WEIGHT * pr_auc - PARSIMONY_WEIGHT * parsimony

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> GAResult:
        """Search for the best feature subset."""
        values = X.to_numpy(dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)

        n_features = values.shape[1]
        population = self._initial_population(n_features, rng)

        history: list[float] = []
        best_mask: np.ndarray | None = None
        best_fitness = -np.inf

        for _ in range(self.generations):
            scores = np.array([self.fitness(m, values, y) for m in population])

            gen_best = int(np.argmax(scores))
            if scores[gen_best] > best_fitness:
                best_fitness = float(scores[gen_best])
                best_mask = population[gen_best].copy()

            history.append(best_fitness)
            population = self._next_generation(population, scores, rng)

        if best_mask is None:
            raise RuntimeError("GA failed to find any viable subset")

        selected = [c for c, keep in zip(X.columns, best_mask) if keep]

        return GAResult(
            mask=best_mask,
            selected=selected,
            fitness=best_fitness,
            pr_auc=_cross_val_pr_auc(
                values[:, best_mask.astype(bool)], y,
                folds=self.cv_folds, random_state=self.random_state,
            ),
            n_selected=len(selected),
            generations=self.generations,
            history=history,
        )

    def _initial_population(self, n_features: int, rng) -> np.ndarray:
        """Every individual starts with at least ``min_features`` bits set."""
        floor = min(self.min_features, n_features)
        population = np.zeros((self.population_size, n_features), dtype=bool)
        for row in population:
            row[:] = False
            row[rng.choice(n_features, size=floor, replace=False)] = True
            row |= rng.random(n_features) < 0.15
        return population

    def _next_generation(
        self, population: np.ndarray, scores: np.ndarray, rng
    ) -> np.ndarray:
        """Elitism + tournament selection + uniform crossover + mutation."""
        n_features = population.shape[1]
        floor = min(self.min_features, n_features)
        next_gen = np.zeros_like(population)

        # Elitism: carry the best individual straight through.
        next_gen[0] = population[int(np.argmax(scores))]

        for i in range(1, self.population_size):
            parent_a = self._tournament(population, scores, rng)
            parent_b = self._tournament(population, scores, rng)

            if rng.random() < self.crossover_rate:
                swap = rng.random(n_features) < 0.5
                child = np.where(swap, parent_a, parent_b)
            else:
                child = parent_a.copy()

            child = child ^ (rng.random(n_features) < self.mutation_rate)

            # Never let a child collapse below the minimum viable size.
            if child.sum() < floor:
                child[:] = False
                child[rng.choice(n_features, size=floor, replace=False)] = True

            next_gen[i] = child

        return next_gen

    def _tournament(
        self, population: np.ndarray, scores: np.ndarray, rng
    ) -> np.ndarray:
        contenders = rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False,
        )
        winner = contenders[int(np.argmax(scores[contenders]))]
        return population[winner]
