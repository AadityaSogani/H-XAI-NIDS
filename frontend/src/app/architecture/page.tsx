import { PageHeader, Panel, SectionLabel } from "@/components/ui";

export const dynamic = "force-dynamic";

const STAGES = [
  ["01", "Data Cleaning", "78 → 67 columns · 10 constant dropped · Inf repaired · metadata removed"],
  ["02", "Domain Features", "7 physics-based ratios: burstiness, direction asymmetry, entropy"],
  ["03", "Stacking Embedding", "K-Means soft distances + GMM responsibilities per flow"],
  ["04", "Hybrid Selection", "Pearson prune → GA fitness over ANOVA + χ² · 83 → 30 features"],
  ["05", "Cluster-Aware SMOTE", "K-Means-guided synthetic minority rows · 1.23% → balanced"],
  ["06", "Hierarchical Detection", "3-level ensemble: binary gate → family head → anomaly head"],
  ["07", "Dual XAI", "TreeSHAP (global + local) and a self-auditing linear surrogate"],
  ["08", "Operational Evaluation", "PR-AUC, FNR, escalation rate on the untouched test tier"],
];

const LEVELS = [
  {
    level: 1,
    name: "Binary gate",
    what: "Ensemble P(attack) vs benign-calibrated threshold",
    exits: "Benign → final",
    color: "#22c55e",
  },
  {
    level: 2,
    name: "Family head",
    what: "Multi-class family classifier (Bot, PortScan, Brute Force, XSS, SQLi)",
    exits: "Confident match → final · else → L3",
    color: "#22d3ee",
  },
  {
    level: 3,
    name: "Anomaly head",
    what: "IsolationForest score, threshold calibrated at p99.9 of benign flows",
    exits: "On-manifold → forced family · off-manifold → Potential Unseen Attack",
    color: "#fb7185",
  },
];

export default function ArchitecturePage() {
  return (
    <div>
      <PageHeader
        eyebrow="System design"
        title="Architecture"
        subtitle="Eight training stages feed a three-level serving hierarchy; every verdict carries its decision path and an explanation."
      />

      <section>
        <SectionLabel>Training pipeline · CICIDS-2017</SectionLabel>
        <div className="mt-3 space-y-2">
          {STAGES.map(([n, title, detail], i) => (
            <Panel key={n} className="flex items-center gap-4 p-4" hover={i < 8}>
              <span className="font-mono text-sm text-cyan-glow">{n}</span>
              <div className="min-w-0">
                <p className="text-sm font-medium text-ink">{title}</p>
                <p className="truncate text-xs text-ink-faint">{detail}</p>
              </div>
            </Panel>
          ))}
        </div>
      </section>

      <section className="mt-10">
        <SectionLabel>Serving hierarchy · one flow, one decision path</SectionLabel>
        <div className="mt-3 grid gap-4 lg:grid-cols-3">
          {LEVELS.map((l) => (
            <Panel key={l.level} className="relative p-5">
              <div
                className="absolute inset-x-5 top-0 h-px"
                style={{ background: l.color }}
              />
              <p className="font-mono text-xs" style={{ color: l.color }}>
                LEVEL {l.level}
              </p>
              <h3 className="mt-2 text-base font-semibold text-ink">{l.name}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-dim">{l.what}</p>
              <p className="mt-3 border-t border-line pt-3 text-xs text-ink-faint">
                {l.exits}
              </p>
            </Panel>
          ))}
        </div>
        <p className="mt-4 max-w-4xl text-sm leading-relaxed text-ink-dim">
          Two deliberate safeguards: the anomaly threshold is calibrated to the
          p99.9 benign score at fit time (a hard-coded 0 would flag ordinary
          traffic as zero-day), and the anomaly head <em>vetoes</em> over-confident
          family calls on off-manifold rows — without it, tree ensembles confidently
          mislabel novel attacks as the nearest known family. The Level-1 gate check
          is <code className="font-mono text-xs text-cyan-glow">p_attack ≤ t and not anomalous</code>,
          so a zero-day bypasses a closed gate when its anomaly score is extreme.
        </p>
      </section>

      <section className="mt-10">
        <SectionLabel>HTTP API · FastAPI on :8000</SectionLabel>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="text-ink-faint">
                <th className="px-3 py-2 font-medium">Method</th>
                <th className="px-3 py-2 font-medium">Path</th>
                <th className="px-3 py-2 font-medium">Returns</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60 text-ink-dim">
              {[
                ["GET", "/health", "liveness + model_loaded + feature count"],
                ["GET", "/metrics", "training report: PR/ROC, per-class, confusion matrix"],
                ["GET", "/features", "the exact feature contract the model expects"],
                ["POST", "/predict", "hierarchical verdict for one flow"],
                ["POST", "/predict/explain", "verdict + SHAP terms + surrogate coefficients"],
                ["POST", "/predict/batch", "verdicts + summary for a JSON array of rows"],
                ["POST", "/predict/upload", "same, for a raw CSV file (32 MB ceiling)"],
              ].map(([method, path, returns]) => (
                <tr key={path}>
                  <td className="px-3 py-2">
                    <span
                      className="rounded px-1.5 py-0.5 font-mono text-[11px]"
                      style={{
                        color: method === "GET" ? "#22d3ee" : "#a78bfa",
                        background:
                          method === "GET" ? "rgba(34,211,238,0.1)" : "rgba(167,139,250,0.1)",
                      }}
                    >
                      {method}
                    </span>
                  </td>
                  <td className="px-3 py-2 font-mono text-xs text-ink">{path}</td>
                  <td className="px-3 py-2 text-xs">{returns}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-10">
        <SectionLabel>Stack</SectionLabel>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {[
            ["Model", "scikit-learn 1.9 · GradientBoosting + RandomForest + IsolationForest"],
            ["Explain", "shap TreeExplainer (exact) · local weighted-linear surrogate"],
            ["API", "FastAPI + Pydantic v2 · joblib artifact + manifest contract"],
            ["UI", "Next.js 16 (App Router) · React 19 · Tailwind v4 · Recharts"],
          ].map(([k, v]) => (
            <Panel key={k} className="p-4">
              <p className="font-mono text-[11px] uppercase tracking-wide text-ink-faint">{k}</p>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-dim">{v}</p>
            </Panel>
          ))}
        </div>
      </section>
    </div>
  );
}
