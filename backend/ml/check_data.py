"""Stage-1 smoke test: load every shard, clean, split, and print a class table.

Run from the ``backend`` directory::

    python -m ml.check_data
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

# Import correctly whether run as a module or a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ml.data import load_shards
    from ml.splitting import stratified_split
    from ml.taxonomy import DISPLAY_NAMES, AttackClass
else:
    from .data import load_shards
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

    started = time.time()
    bundle = load_shards([Path(p) for p in paths])
    elapsed = time.time() - started

    print(f"Loaded + cleaned {len(paths)} shard(s) in {elapsed:.1f}s")
    print(f"  X shape      : {bundle.X.shape}")
    print(f"  features     : {len(bundle.feature_names)}")
    print(f"  attack rate  : {bundle.report['attack_rate']:.4%}")
    print(f"  constants dropped : {len(bundle.report['constant_columns_dropped'])}")
    print()

    split = stratified_split(bundle.y_family)
    sizes = split.summary()
    print(f"  train={sizes['train']:,}  val={sizes['validation']:,}  test={sizes['test']:,}")
    print()
    print(f"{'class':<18}{'total':>10}{'train':>10}{'val':>8}{'test':>8}")
    print("-" * 54)

    for cls in AttackClass:
        total = int((bundle.y_family == cls).sum())
        if total == 0:
            continue
        tr = int((bundle.y_family[split.train] == cls).sum())
        va = int((bundle.y_family[split.validation] == cls).sum())
        te = int((bundle.y_family[split.test] == cls).sum())
        print(f"{DISPLAY_NAMES[cls]:<18}{total:>10,}{tr:>10,}{va:>8,}{te:>8,}")

    split.assert_disjoint()
    print("\n  disjoint + exhaustive: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
