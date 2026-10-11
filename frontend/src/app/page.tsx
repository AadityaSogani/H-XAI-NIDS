import Link from "next/link";
import { ErrorBanner, Panel, SectionLabel, StatusDot } from "@/components/ui";
import { api } from "@/lib/api";

/**
 * Landing page.
 *
 * Fetches live health and metrics so the headline numbers are the real trained
 * model's, not hard-coded claims.  When the backend is unreachable the page
 * still renders and says what to start — a page that 500s when a service is
 * down is worse than one that degrades honestly.
 */
export const dynamic = "force-dynamic";

const PIPELINE = [
  { stage: "01", title: "Data Cleaning", detail: "78 → 67 columns · 10 constant features removed" },
  { stage: "02", title: "Domain Features", detail: "7 physics-based ratios · burstiness, asymmetry, entropy" },
  { stage: "03", title: "Stacking Embedding", detail: "K-Means soft distances + GMM responsibilities" },
  { stage: "04", title: "Hybrid Selection", detail: "Pearson → ANOVA → χ² · 83 → 30 features" },
  { stage: "05", title: "Cluster-Aware SMOTE", detail: "k-Means SMOTE · 1.23% attack rate rebalanced" },
  { stage: "06", title: "Hierarchical Detection", detail: "L1 binary · L2 family · L3 zero-day" },
  { stage: "07", title: "Dual XAI", detail: "Global TreeSHAP + local surrogate" },
  { stage: "08", title: "Operational Eval", detail: "PR-AUC, FNR, escalation rate" },
];

export default async function HomePage() {
  let health: Awaited<ReturnType<typeof api.health>> | null = null;
  let metrics: Awaited<ReturnType<typeof api.metrics>> | null = null;
  let error: string | null = null;

  try {
    [health, metrics] = await Promise.all([api.health(), api.metrics()]);
  } catch (err) {
    error = err instanceof Error ? err.message : "Backend unreachable";
  }

  const level1 = metrics?.metrics?.level1;
  const familyAcc = metrics?.metrics?.level2_family_accuracy_on_attacks;

  return (
    <div className="space-y-10">
      <Hero error={error} health={health} />
      {error && (
        <ErrorBanner
          message={`${error} — start the API with: uvicorn app.main:app --port 8000`}
        />
      )}
      <Metrics level1={level1} familyAcc={familyAcc} />
      <Pipeline />
      <Limitations />
    </div>
  );

function Hero({
  error,
  health,
}: {
  error: string | null;
  health: Awaited<ReturnType<typeof api.health>> | null;
}) {
  return (
    <section className="aurora relative overflow-hidden rounded-2xl border border-line px-6 py-14 sm:px-10">
      <div className="grid-backdrop absolute inset-0" aria-hidden />
      <div className="relative z-10 max-w-3xl">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-cyan-glow">
          Hierarchical Contextual Stacking · CICIDS-2017
        </p>
        <h1 className="mt-4 text-4xl font-semibold leading-tight tracking-tight sm:text-5xl">
          Network intrusion detection that{" "}
          <span className="bg-gradient-to-r from-cyan-glow to-violet-glow bg-clip-text text-transparent">
            explains itself
          </span>
        </h1>
        <p className="mt-5 text-base leading-relaxed text-ink-dim">
          A three-level detector separates <em>whether</em> traffic is hostile
          from <em>which family</em> it belongs to, then routes uncertain flows
          to an anomaly head that flags potential zero-day attacks instead of
          forcing them into a known label. Every alert carries a TreeSHAP
          attribution, so an analyst sees why — not just what.
        </p>

        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/detect"
            className="rounded-lg border border-cyan-glow/50 bg-cyan-glow/10 px-5 py-2.5 text-sm font-medium text-cyan-glow transition hover:bg-cyan-glow/20"
          >
            Analyse a flow →
          </Link>
          <Link
            href="/dashboard"
            className="rounded-lg border border-line bg-surface-2/60 px-5 py-2.5 text-sm text-ink-dim transition hover:border-violet-glow/40 hover:text-ink"
          >
            View model performance
          </Link>
        </div>

        <div className="mt-7">
          {error ? (
            <StatusDot ok={false} label="Detection API offline" />
          ) : (
            <StatusDot
              ok={health?.model_loaded ?? null}
              label={
                health?.model_loaded
                  ? `Model live · ${health.n_features} features`
                  : "API up · no model loaded"
              }
            />
          )}
        </div>
      </div>
    </section>
  );
}

function Metrics({
  level1,
  familyAcc,
}: {
  level1: { pr_auc: number; roc_auc: number; attack_rate: number } | undefined;
  familyAcc: number | undefined;
}) {
  return (
    <section>
      <SectionLabel>Measured on the held-out test tier</SectionLabel>
      <div className="mt-3 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric
          label="PR-AUC"
          value={level1 ? level1.pr_auc.toFixed(4) : "—"}
          note="primary metric — honest under a 1.23% attack rate"
          accent="#22d3ee"
        />
        <Metric
          label="ROC-AUC"
          value={level1 ? level1.roc_auc.toFixed(4) : "—"}
          note="secondary, majority-dominated"
          accent="#a78bfa"
        />
        <Metric
          label="Attack rate"
          value={level1 ? `${(level1.attack_rate * 100).toFixed(2)}%` : "—"}
          note="5,536 attacks / 451,380 flows"
          accent="#f97316"
        />
        <Metric
          label="Family accuracy"
          value={familyAcc != null ? `${(familyAcc * 100).toFixed(1)}%` : "—"}
          note="on true attacks, excluding escalations"
          accent="#22c55e"
        />
      </div>
    </section>
  );
}

function Metric({
  label,
  value,
  note,
  accent,
}: {
  label: string;
  value: string;
  note: string;
  accent: string;
}) {
  return (
    <Panel className="p-4">
      <p className="font-mono text-[11px] uppercase tracking-wider text-ink-faint">
        {label}
      </p>
      <p
        className="mt-2 font-mono text-3xl font-semibold tabular-nums"
        style={{ color: accent }}
      >
        {value}
      </p>
      <p className="mt-1.5 text-xs leading-snug text-ink-faint">{note}</p>
    </Panel>
  );
}

function Pipeline() {
  return (
    <section>
      <SectionLabel>Eight-stage pipeline</SectionLabel>
      <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {PIPELINE.map((step) => (
          <Panel key={step.stage} hover className="p-4">
            <p className="font-mono text-xs text-cyan-glow">{step.stage}</p>
            <h3 className="mt-1.5 text-sm font-medium text-ink">{step.title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-ink-faint">
              {step.detail}
            </p>
          </Panel>
        ))}
      </div>
    </section>
  );
}

function Limitations() {
  return (
    <Panel className="p-6">
      <SectionLabel>Known limitations</SectionLabel>
      <p className="mt-2 max-w-4xl text-sm leading-relaxed text-ink-dim">
        Reported as-is rather than buried:{" "}
        <strong className="text-ink">
          Brute Force and XSS are frequently confused
        </strong>{" "}
        (F1 0.68 and 0.40) because both are &ldquo;Web Attack&rdquo; subtypes
        drawn from the same Thursday capture and share a feature profile.{" "}
        <strong className="text-ink">
          SQL Injection has zero test support
        </strong>{" "}
        — only 21 rows exist corpus-wide, so it is held entirely in training and
        cannot be scored honestly. No superiority over published baselines is
        claimed without a controlled comparison under a common protocol.
      </p>
    </Panel>
  );
}

}
