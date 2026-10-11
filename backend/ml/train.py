"""End-to-end training run for the H-XAI-NIDS hierarchy.

Executes the full pipeline on the real CICIDS-2017 shards and prints the
metrics that matter for a 1.23%-attack corpus.  PR-AUC is reported alongside
ROC-AUC because ROC-AUC is dominated by the benign majority and looks healthy
even for a classifier that misses every attack.

Run from the ``backend`` directory::

    python -m ml.train
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.balancing import KMeansSMOTE
    from ml.data import apply_log1p, load_shards
    from ml.features import add_domain_features
    from ml.hierarchy import HierarchicalDetector
    from ml.selection import hybrid_select
    from ml.sfe import SFETransformer
    from ml.splitting import stratified_split
    from ml.taxonomy import AttackClass
else:
    from .balancing import KMeansSMOTE
    from .data import apply_log1p, load_shards
    from .features import add_domain_features
    from .hierarchy import HierarchicalDetector
    from .selection import hybrid_select
    from .sfe import SFETransformer
    from .splitting import stratified_split
    from .taxonomy import AttackClass

DATA_GLOB = (
    "../Network-Detection-Intrusion-System/Datasets/Sogani Datasets/*.parquet"
)
ARTIFACT_DIR = Path("artifacts")


def concat_columns(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Column-wise concat that tolerates a reset index on either side."""
    return pd.concat([a.reset_index(drop=True), b.reset_index(drop=True)], axis=1)


def main() -> int:
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No parquet shards matched {DATA_GLOB!r}")
        return 1

    ARTIFACT_DIR.mkdir(exist_ok=True)
    report: dict = {}

    # --- Stage 1: clean ----------------------------------------------------
    started = time.time()
    bundle = load_shards([Path(p) for p in paths])
    report["stage1"] = bundle.report
    print(f"[1/6] cleaned {bundle.X.shape[0]:,} rows x {bundle.X.shape[1]} cols "
          f"({time.time() - started:.1f}s)")

    # --- Stage 2-3: features (SFE fitted on TRAIN rows only) ---------------
    started = time.time()
    X, domain_cols = add_domain_features(bundle.X)
    X = apply_log1p(X)

    split = stratified_split(bundle.y_family)
    X_train_raw = X.iloc[split.train].reset_index(drop=True)
    y_train = bundle.y_family[split.train]

    sfe = SFETransformer()
    X_train = concat_columns(X_train_raw, sfe.fit_transform(X_train_raw))
    report["features"] = {
        "raw": int(bundle.X.shape[1]),
        "domain": len(domain_cols),
        "sfe": sfe.n_features_out,
        "total": int(X_train.shape[1]),
    }
    print(f"[2/6] features: {X_train.shape[1]} candidates "
          f"({bundle.X.shape[1]} raw + {len(domain_cols)} domain + "
          f"{sfe.n_features_out} SFE) in {time.time() - started:.1f}s")

    # --- Stage 4: selection ------------------------------------------------
    started = time.time()
    selection = hybrid_select(X_train, y_train, shortlist_size=30)
    features = selection.selected
    report["selection"] = selection.summary()
    print(f"[3/6] selected {len(features)} of {X_train.shape[1]} features "
          f"(dropped {len(selection.pearson_dropped)} collinear) "
          f"in {time.time() - started:.1f}s")

    X_sel = X_train[features].to_numpy(dtype=np.float64)

    # --- Stage 5: balancing ------------------------------------------------
    started = time.time()
    balancer = KMeansSMOTE(ratio=0.3, random_state=42)
    X_bal, y_bal = balancer.fit_resample(X_sel, y_train)
    report["balancing"] = {
        "before": {int(c): int((y_train == c).sum()) for c in np.unique(y_train)},
        "after": {int(c): int((y_bal == c).sum()) for c in np.unique(y_bal)},
        "config": balancer.describe(),
    }
    print(f"[4/6] balanced {len(y_train):,} -> {len(y_bal):,} rows "
          f"in {time.time() - started:.1f}s")

    # --- Stage 6: hierarchy ------------------------------------------------
    started = time.time()
    model = HierarchicalDetector(random_state=42).fit(X_bal, y_bal)
    report["training_seconds"] = round(time.time() - started, 1)
    print(f"[5/6] trained 3-level hierarchy in {time.time() - started:.1f}s")
    print(f"      anomaly threshold calibrated at "
          f"{model.unseen_threshold:.4f} (p99.9 of benign)")

    # --- Evaluate on the untouched test tier -------------------------------
    print("\n[6/6] evaluation on the held-out test tier")
    X_test_raw = X.iloc[split.test].reset_index(drop=True)
    X_test = concat_columns(X_test_raw, sfe.transform(X_test_raw))
    X_test = X_test[features].to_numpy(dtype=np.float64)
    y_test = bundle.y_family[split.test]

    y_bin_true = (y_test != int(AttackClass.BENIGN)).astype(int)
    p_attack = model.predict_proba_attack(X_test)

    report["level1"] = {
        "pr_auc": round(float(average_precision_score(y_bin_true, p_attack)), 4),
        "roc_auc": round(float(roc_auc_score(y_bin_true, p_attack)), 4),
        "attack_rate": round(float(y_bin_true.mean()), 5),
        "rows": int(len(y_test)),
    }
    print(f"  Level-1 (binary)   PR-AUC {report['level1']['pr_auc']:.4f}   "
          f"ROC-AUC {report['level1']['roc_auc']:.4f}")

    verdicts = model.predict(X_test)

    # Family accuracy over true attacks only, excluding unseen escalations.
    attack_pos = np.flatnonzero(y_bin_true == 1)
    correct = sum(
        1 for i in attack_pos
        if verdicts[i]["label_id"] == int(y_test[i]) and not verdicts[i]["unseen"]
    )
    report["level2_family_accuracy_on_attacks"] = round(
        correct / max(len(attack_pos), 1), 4
    )
    report["attacks_escalated_as_unseen"] = sum(
        1 for i in attack_pos if verdicts[i]["unseen"]
    )
    report["benign_escalated_as_unseen"] = sum(
        1 for i in np.flatnonzero(y_bin_true == 0) if verdicts[i]["unseen"]
    )
    report["level_distribution"] = {
        str(lvl): sum(1 for v in verdicts if v["level"] == lvl) for lvl in (1, 2, 3)
    }

    print(f"  Level-2 family acc on true attacks: "
          f"{report['level2_family_accuracy_on_attacks']:.2%}")
    print(f"  Level-3 escalations: {report['attacks_escalated_as_unseen']:,} attacks "
          f"+ {report['benign_escalated_as_unseen']:,} benign rows called unseen")

    # Persisted into training_report.json so the dashboard charts the real
    # evaluation rather than hard-coded figures.
    report["evaluation"] = _evaluate(y_test, y_bin_true, p_attack, verdicts)
    _print_classification(report["evaluation"])

    # --- Persist the serving artifact ---------------------------------------
    from ml.registry import NIDSArtifact

    # A stratified background sample drives SHAP and the local surrogate at
    # serving time.  Drawn from the test tier so explanations are not computed
    # on the very rows the model was fitted to.
    background_idx = np.random.default_rng(42).choice(
        len(X_test), size=min(1500, len(X_test)), replace=False
    )
    artifact = NIDSArtifact(
        model=model,
        feature_names=features,
        metadata={
            "version": "1.0.0",
            "selected_features": features,
            "level1": report["level1"],
            "level2_family_accuracy": report["level2_family_accuracy_on_attacks"],
            "anomaly_threshold": model.unseen_threshold,
            "train_rows": int(len(y_train)),
            "balanced_rows": int(len(y_bal)),
        },
    )
    artifact.save(ARTIFACT_DIR)

    background_path = ARTIFACT_DIR / "background.npy"
    np.save(background_path, X_test[background_idx])

    (ARTIFACT_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\n  artifact written to {ARTIFACT_DIR}/ (model + manifest + background)")
    return 0


def _evaluate(
    y_test: np.ndarray,
    y_bin_true: np.ndarray,
    p_attack: np.ndarray,
    verdicts: list[dict],
) -> dict:
    """Per-class metrics, confusion matrix and downsampled PR/ROC curves.

    Everything here is persisted into training_report.json so the dashboard
    charts real numbers.  Curves are capped at 100 points (every k-th) —
    full-resolution curves would be megabytes for little visual gain.
    """
    pred_ids = np.array(
        [v["label_id"] if v["label_id"] >= 0 else 99 for v in verdicts]
    )
    present = sorted({int(c) for c in y_test.tolist()} | set(pred_ids.tolist()))

    names = []
    for c in present:
        if c == 99:
            names.append("UNSEEN")
        else:
            # AttackClass.name matches the verdict labels the serving hierarchy
            # emits (BENIGN, PORT_SCAN, XSS, …) so one colour ramp works across
            # charts, badges and tables.
            names.append(AttackClass(c).name)

    report_dict = classification_report(
        y_test, pred_ids, labels=present, target_names=names,
        zero_division=0, output_dict=True,
    )
    per_class = {
        name: {
            "precision": round(report_dict[name]["precision"], 4),
            "recall": round(report_dict[name]["recall"], 4),
            "f1": round(report_dict[name]["f1-score"], 4),
            "support": int(report_dict[name]["support"]),
        }
        for name in names
    }

    cm = confusion_matrix(y_test, pred_ids, labels=present)

    def _sample(*arrays: np.ndarray) -> list[list[float]]:
        """Evenly subsample curve points and round to 4 decimals."""
        n = max(len(a) for a in arrays)
        idx = np.unique(np.linspace(0, n - 1, min(n, 100)).astype(int))
        return [a[idx].round(4).tolist() for a in arrays]

    pr_precision, pr_recall, _ = precision_recall_curve(y_bin_true, p_attack)
    roc_fpr, roc_tpr, _ = roc_curve(y_bin_true, p_attack)
    pr_p, pr_r = _sample(pr_precision, pr_recall)
    rc_f, rc_t = _sample(roc_fpr, roc_tpr)

    return {
        "per_class": per_class,
        "confusion_matrix": {"labels": names, "matrix": cm.tolist()},
        "curves": {
            "pr": {"precision": pr_p, "recall": pr_r},
            "roc": {"fpr": rc_f, "tpr": rc_t},
        },
    }


def _print_classification(evaluation: dict) -> None:
    """Console mirror of the persisted evaluation (UNSEEN as its own class)."""
    per_class = evaluation["per_class"]
    cm = evaluation["confusion_matrix"]

    print("\n  per-class report (test tier):")
    print(f"   {'class':>12} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for name, m in per_class.items():
        print(
            f"   {name:>12} {m['precision']:>10.3f} {m['recall']:>10.3f} "
            f"{m['f1']:>10.3f} {m['support']:>10,}"
        )

    print("  confusion matrix (rows = true, cols = predicted):")
    labels = cm["labels"]
    print("   " + " " * 10 + " ".join(f"{n[:9]:>10}" for n in labels))
    for name, row in zip(labels, cm["matrix"]):
        print(f"   {name[:10]:>10} " + " ".join(f"{v:>10,}" for v in row))


if __name__ == "__main__":
    raise SystemExit(main())
