"""Stage-4 smoke test: how much redundancy does hybrid selection remove?

Run from the ``backend`` directory::

    python -m ml.check_selection
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.data import apply_log1p, load_shards
    from ml.features import add_domain_features
    from ml.selection import hybrid_select
    from ml.sfe import SFETransformer
    from ml.splitting import stratified_split
else:
    from .data import apply_log1p, load_shards
    from .features import add_domain_features
    from .selection import hybrid_select
    from .sfe import SFETransformer
    from .splitting import stratified_split

DATA_GLOB = (
    "../Network-Detection-Intrusion-System/Datasets/Sogani Datasets/*.parquet"
)


def main() -> int:
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No parquet shards matched {DATA_GLOB!r}")
        return 1

    bundle = load_shards([Path(p) for p in paths])
    X, domain_cols = add_domain_features(bundle.X)
    X = apply_log1p(X)

    split = stratified_split(bundle.y_family)
    X_train = X.iloc[split.train]
    y_train = bundle.y_family[split.train]

    print(f"candidates : {X.shape[1]} columns ({len(bundle.feature_names)} raw "
          f"+ {len(domain_cols)} domain)")

    # SFE is fitted on train only, then applied everywhere.
    started = time.time()
    sfe = SFETransformer()
    emb_train = sfe.fit_transform(X_train)
    X_train_full = pd.concat([X_train.reset_index(drop=True), emb_train], axis=1)
    print(f"SFE        : +{emb_train.shape[1]} columns, fitted on "
          f"{len(X_train):,} train rows in {time.time() - started:.1f}s")

    started = time.time()
    result = hybrid_select(X_train_full, y_train, shortlist_size=30)
    elapsed = time.time() - started

    print(f"\nhybrid selection in {elapsed:.1f}s")
    print(f"  Pearson dropped : {len(result.pearson_dropped)} collinear columns")
    print(f"  shortlisted     : {len(result.selected)} of "
          f"{X_train_full.shape[1] - len(result.pearson_dropped)} survivors")
    print(f"  final features  : {len(result.selected)} "
          f"({len(result.selected) / X.shape[1]:.0%} of the {X.shape[1]} raw candidates)")

    print("\n  strongest collinear pairs (|r|):")
    for a, b, r in result.pearson_pairs[:12]:
        print(f"    {r:.4f}  {a}  ~  {b}")

    print("\n  top 30 selected features by blended ANOVA+chi2 rank:")
    ranked = sorted(result.selected, key=lambda c: result.anova_scores.get(c, 0.0), reverse=True)
    for i, col in enumerate(ranked, 1):
        tag = "domain" if col in domain_cols else ("SFE" if col.startswith("SFE_") else "raw")
        print(f"    {i:2d}. {col:<32s} F={result.anova_scores.get(col, 0.0):>12.1f}  [{tag}]")

    derived_kept = [c for c in result.selected if c in domain_cols or c.startswith("SFE_")]
    print(f"\n  derived features surviving selection: {len(derived_kept)}")
    for c in derived_kept:
        print(f"    - {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
