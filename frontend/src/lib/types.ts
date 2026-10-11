/**
 * Types mirroring the FastAPI response contracts in backend/app/schemas.py.
 *
 * Kept hand-written rather than generated: the surface is small and stable, and
 * a checked-in copy means the UI compiles even when the backend is not running.
 */

export interface Verdict {
  label: string;
  label_id: number;
  is_attack: boolean;
  unseen: boolean;
  /** Which level produced the decision: 1 = binary, 2 = family, 3 = zero-day. */
  level: 1 | 2 | 3;
  confidence: number;
  p_attack: number;
  anomaly_score: number;
}

export interface Attribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface Explanation {
  verdict: Verdict;
  shap: Attribution[];
  surrogate_fidelity_r2: number | null;
  surrogate: Attribution[] | null;
}

export interface Prediction {
  verdict: Verdict;
  explanation: Explanation | null;
  model_version: string | null;
}

export interface BatchResult {
  total: number;
  attacks: number;
  unseen: number;
  verdicts: Verdict[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  n_features: number | null;
  artifact_dir: string;
  version: string;
}

export interface FeatureContract {
  n_features: number;
  feature_names: string[];
  metadata: Record<string, unknown>;
}

export interface TrainingMetrics {
  available: boolean;
  metrics: {
    stage1?: Record<string, unknown>;
    features?: Record<string, number>;
    selection?: Record<string, unknown>;
    balancing?: Record<string, unknown>;
    level1?: {
      pr_auc: number;
      roc_auc: number;
      attack_rate: number;
      rows: number;
    };
    level2_family_accuracy_on_attacks?: number;
    attacks_escalated_as_unseen?: number;
    benign_escalated_as_unseen?: number;
    level_distribution?: Record<string, number>;
    training_seconds?: number;
  } | null;
  message?: string | null;
}

/** Display metadata per attack family, shared by every chart and badge. */
export const FAMILY_STYLE: Record<
  string,
  { color: string; short: string; severity: number }
> = {
  BENIGN: { color: "#22c55e", short: "Benign", severity: 0 },
  BOT: { color: "#f97316", short: "Bot", severity: 3 },
  PORT_SCAN: { color: "#eab308", short: "Port Scan", severity: 2 },
  BRUTE_FORCE: { color: "#f43f5e", short: "Brute Force", severity: 4 },
  XSS: { color: "#a78bfa", short: "XSS", severity: 4 },
  SQL_INJECTION: { color: "#ec4899", short: "SQL Injection", severity: 5 },
  "Potential Unseen Attack": { color: "#fb7185", short: "Unseen", severity: 5 },
};

export function styleFor(label: string) {
  return (
    FAMILY_STYLE[label] ?? {
      color: "#94a3b8",
      short: label,
      severity: 1,
    }
  );
}
