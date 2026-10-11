"""Do the derived features actually separate attacks from benign traffic?

Prints per-class means and a one-way ANOVA F-statistic for every derived
feature.  A high F-statistic means the feature carries class information that
the raw aggregates miss.

Run from the ``backend`` directory::

    python -m ml.check_features
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.data import load_shards
    from ml.features import DOMAIN_FEATURES, add_domain_features
    from ml.sfe import SFETransformer
    from ml.splitting import stratified_split
    from ml.taxonomy import DISPLAY_NAMES, AttackClass
else:
    from .data import load_shards
    from .features import DOMAIN_FEATURES, add_domain_features
    from .sfe import SFETransformer
    from .splitting import stratified_split
    from .taxonomy import DISPLAY_NAMES, AttackClass

DATA_GLOB = (
    "../Network-Detection-Intrusion-System/Datasets/Sogani Datasets/*.parquet"
)


def main() -> int:
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No parquet shards matched {DATA_GLOB!r}")
        return 1

    bundle = load_shards([Path(p) for p in paths])
    X, new_cols = add_domain_features(bundle.X)
    print(f"features: {bundle.X.shape[1]} raw -> {X.shape[1]} with domain features\n")

    split = stratified_split(bundle.y_family)
    present = [c for c in AttackClass if (bundle.y_family == c).sum() > 0]

    print(f"{'feature':<28}", end="")
    for cls in present:
        print(f"{DISPLAY_NAMES[cls][:11]:>13}", end="")
    print(f"{'F-stat':>12}")
    print("-" * (28 + 13 * len(present) + 12))

    derived = X[new_cols]
    for col in new_cols:
        values = derived[col].to_numpy()
        print(f"{col:<28}", end="")
        groups = []
        for cls in present:
            mask = bundle.y_family == cls
            groups.append(values[mask])
            print(f"{values[mask].mean():>13.4f}", end="")

        # One-way ANOVA across the non-benign families vs benign.
        overall = values.mean()
        between = sum(len(g) * (g.mean() - overall) ** 2 for g in groups)
        within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        k = len(groups)
        f = (between / (k - 1)) / (within / (len(values) - k)) if within > 0 else np.inf
        print(f"{f:>12.1f}")

    # --- SFE sanity check, fitted on train only ---
    print("\nSFE (fitted on train rows only):")
    sfe = SFETransformer()
    train_rows = X.iloc[split.train]
    emb_train = sfe.fit_transform(train_rows)
    print(f"  produced {emb_train.shape[1]} columns: {list(emb_train.columns)}")
    print(f"  finite: {np.isfinite(emb_train.to_numpy()).all()}")

    # A single held-out row must transform without error (serving path).
    single = sfe.transform(X.iloc[split.test[:1]])
    print(f"  single-row transform OK: shape={single.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
