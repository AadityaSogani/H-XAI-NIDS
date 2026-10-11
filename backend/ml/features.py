"""Stage 2 — domain-specific feature engineering.

The raw CICIDS-2017 columns are *aggregate* flow statistics (totals, means,
standard deviations).  They describe how much traffic flowed but not the *shape*
of that traffic.  An attacker and a legitimate bulk transfer can produce nearly
identical packet counts while differing sharply in rhythm and directionality.

The seven features below encode those behavioural properties explicitly.  They
are all ratios/coefficients of existing columns, so they add no new measurement
— only new *representation*.

Every function is vectorised and division-safe: a zero denominator yields 0
rather than NaN/inf, because a zero-length flow is a real and meaningful case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Names of the derived features, in a fixed order so artifacts stay aligned.
DOMAIN_FEATURES = [
    "Payload_Ratio_Fwd",
    "Byte_Asymmetry_Index",
    "Packet_Rate_Asymmetry",
    "IAT_Coefficient_Variation",
    "Burstiness_Index",
    "Pkt_Size_Skewness",
    "Byte_Entropy",
]

_EPS = 1e-9


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` returning 0 where the denominator is ~0."""
    den = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)
    out = np.zeros_like(num, dtype=np.float64)
    np.divide(num, den, out=out, where=np.abs(den) > _EPS)
    return out


def _col(X: pd.DataFrame, name: str) -> np.ndarray:
    """Fetch a column as float64, or zeros if absent.

    Absent columns are tolerated so the same code runs on reduced feature sets
    (e.g. after selection) without special-casing every caller.
    """
    if name in X.columns:
        return X[name].to_numpy(dtype=np.float64)
    return np.zeros(len(X), dtype=np.float64)


def add_domain_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Append the seven domain features, returning ``(X_out, new_columns)``."""
    if len(X) == 0:
        return X.copy(), []

    fwd_bytes = _col(X, "Fwd Packets Length Total")
    bwd_bytes = _col(X, "Bwd Packets Length Total")
    fwd_pkts = _col(X, "Total Fwd Packets")
    bwd_pkts = _col(X, "Total Backward Packets")
    iat_mean = _col(X, "Flow IAT Mean")
    iat_std = _col(X, "Flow IAT Std")
    iat_total = _col(X, "Fwd IAT Total")
    iat_min = _col(X, "Flow IAT Min")
    pkt_mean = _col(X, "Packet Length Mean")
    pkt_std = _col(X, "Packet Length Std")
    fwd_rate = _col(X, "Fwd Packets/s")
    bwd_rate = _col(X, "Bwd Packets/s")
    pkt_len_max = _col(X, "Packet Length Max")
    pkt_len_min = _col(X, "Packet Length Min")

    # 1. How much of the payload travels forward?  Exfiltration and injection
    #    push data one way; interactive traffic is more balanced.
    payload_ratio = _safe_ratio(fwd_bytes, fwd_bytes + bwd_bytes)

    # 2. Signed directional imbalance, normalised to [-1, 1].  2*p - 1 maps the
    #    forward fraction onto a signed scale where 0 means perfectly balanced.
    byte_asymmetry = 2.0 * payload_ratio - 1.0

    # 3. Same idea for packet *rates* rather than bytes.  Rate asymmetry catches
    #    asymmetric floods that byte asymmetry misses (many tiny packets).
    rate_sum = fwd_rate + bwd_rate
    rate_asymmetry = 2.0 * _safe_ratio(fwd_rate, rate_sum) - 1.0

    # 4. Coefficient of variation of inter-arrival times.  Low CV = machine-like
    #    regularity (scripted flood); high CV = bursty human traffic.
    iat_cv = _safe_ratio(iat_std, np.abs(iat_mean))

    # 5. Burstiness relative to the flow's own mean gap.  A flow concentrated in
    #    a few bursts has IAT_Total much larger than its mean implies.
    burstiness = _safe_ratio(iat_total - iat_min, np.abs(iat_mean))

    # 6. (mean - min) / (max - min): where the bulk of packets sit in the size
    #    range.  Near 1 => uniformly large (bulk transfer); near 0 => many small
    #    packets with a few large ones (scan or covert channel).
    pkt_span = pkt_len_max - pkt_len_min
    pkt_skewness = _safe_ratio(pkt_mean - pkt_len_min, pkt_span)

    # 7. Shannon entropy of the forward/backward byte split, in bits.  Peaks at
    #    1 bit for a balanced flow and falls to 0 for a one-way transfer.
    p = np.clip(payload_ratio, 0.0, 1.0)
    byte_entropy = -(
        p * np.log2(np.clip(p, _EPS, 1.0))
        + (1.0 - p) * np.log2(np.clip(1.0 - p, _EPS, 1.0))
    )

    derived = pd.DataFrame(
        {
            "Payload_Ratio_Fwd": payload_ratio,
            "Byte_Asymmetry_Index": byte_asymmetry,
            "Packet_Rate_Asymmetry": rate_asymmetry,
            "IAT_Coefficient_Variation": iat_cv,
            "Burstiness_Index": burstiness,
            "Pkt_Size_Skewness": pkt_skewness,
            "Byte_Entropy": byte_entropy,
        },
        index=X.index,
    )

    # Final safety net: no derived feature may be NaN or infinite.
    derived = derived.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return pd.concat([X, derived], axis=1), list(DOMAIN_FEATURES)


def describe_domain_features() -> list[dict[str, str]]:
    """Human-readable descriptions, reused by the docs page and reports."""
    return [
        {
            "name": "Payload_Ratio_Fwd",
            "formula": "FwdBytes / (FwdBytes + BwdBytes)",
            "captures": "Direction of payload transfer",
        },
        {
            "name": "Byte_Asymmetry_Index",
            "formula": "2 * Payload_Ratio_Fwd - 1",
            "captures": "Signed directional imbalance, [-1, 1]",
        },
        {
            "name": "Packet_Rate_Asymmetry",
            "formula": "2 * FwdRate/(FwdRate+BwdRate) - 1",
            "captures": "Rate asymmetry invisible to byte totals",
        },
        {
            "name": "IAT_Coefficient_Variation",
            "formula": "IAT_Std / |IAT_Mean|",
            "captures": "Machine-like regularity vs human burstiness",
        },
        {
            "name": "Burstiness_Index",
            "formula": "(IAT_Total - IAT_Min) / |IAT_Mean|",
            "captures": "Traffic concentrated into few bursts",
        },
        {
            "name": "Pkt_Size_Skewness",
            "formula": "(Mean - Min) / (Max - Min)",
            "captures": "Bulk transfer vs scan/covert channel",
        },
        {
            "name": "Byte_Entropy",
            "formula": "-p log2 p - (1-p) log2 (1-p)",
            "captures": "One-way transfer vs balanced exchange",
        },
    ]

    return np.zeros(len(X), dtype=np.float64)
