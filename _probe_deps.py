"""Phase 0 dependency compatibility probe.

Verifies that the bleeding-edge stack (Python 3.13 / numpy 2.5 / pandas 3.0)
can import AND run the risky dependencies that H-XAI-NIDS depends on:
shap (numba-backed), imbalanced-learn, xgboost, lightgbm.

Run:  python _probe_deps.py
"""
import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probe(name):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:<16} OK      version={ver}")
        return mod
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<16} FAIL    {type(exc).__name__}: {exc}")
        return None


print("=== import probe ===")
shap = probe("shap")
imblearn = probe("imblearn")
xgb = probe("xgboost")
lgb = probe("lightgbm")
probe("sklearn")

print("\n=== functional probe (small synthetic run) ===")
import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 8))
y = (X[:, 0] + rng.normal(scale=0.3, size=200) > 0).astype(int)

rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
print("  RandomForest          OK      (control baseline)")

# --- SHAP: the highest-risk dependency (numba must build on 3.13) ---
if shap is not None:
    try:
        expl = shap.TreeExplainer(rf)
        arr = np.asarray(expl.shap_values(X[:20]))
        print(f"  shap.TreeExplainer    OK      shap_values shape={arr.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  shap.TreeExplainer    FAIL    {type(exc).__name__}: {exc}")

# --- imbalanced-learn: k-Means SMOTE is the planned balancing strategy ---
if imblearn is not None:
    try:
        from imblearn.over_sampling import SMOTE

        Xr, yr = SMOTE(k_neighbors=3, random_state=0).fit_resample(X, y)
        print(f"  imblearn SMOTE        OK      {X.shape} -> {Xr.shape}")
    except Exception as exc:  # noqa: BLE001
        print(f"  imblearn SMOTE        FAIL    {type(exc).__name__}: {exc}")

# --- gradient boosting baselines ---
if xgb is not None:
    try:
        m = xgb.XGBClassifier(n_estimators=25, max_depth=3, verbosity=0)
        m.fit(X, y)
        print(f"  xgboost XGBClassifier  OK     train_acc={m.score(X, y):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  xgboost XGBClassifier  FAIL   {type(exc).__name__}: {exc}")

if lgb is not None:
    try:
        m = lgb.LGBMClassifier(n_estimators=25, verbose=-1)
        m.fit(X, y)
        print(f"  lightgbm LGBMClassifier OK    train_acc={m.score(X, y):.3f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  lightgbm LGBMClassifier FAIL  {type(exc).__name__}: {exc}")

print("\n=== probe complete ===")
