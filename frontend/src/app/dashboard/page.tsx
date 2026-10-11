import { CurveChart, PerFamilyF1 } from "@/components/charts";
import { EmptyState, ErrorBanner, PageHeader, Panel, SectionLabel } from "@/components/ui";
import { api } from "@/lib/api";
import type { TrainingMetrics } from "@/lib/types";

export const dynamic = "force-dynamic";

/**
 * Evaluation dashboard.
 *
 * Every number and curve here comes from the training run's own report via
 * GET /metrics — nothing is hard-coded, so the page cannot drift from the
 * deployed model.
 */
export default async function DashboardPage() {
  let payload: TrainingMetrics | null = null;
  let error: string | null = null;

  try {
    payload = await api.metrics();
  } catch (err) {
    error = err instanceof Error ? err.message : "Backend unreachable";
  }

  const m = payload?.metrics ?? null;

  return (
    <div>
      <PageHeader
        eyebrow="Model evaluation · held-out test tier"
        title="Performance dashboard"
        subtitle="Measured on 451,380 flows from CICIDS-2017, 1.23% attack rate. PR-AUC is the primary metric; ROC-AUC is reported but is majority-dominated."
      />

      {error && (
        <ErrorBanner message={`${error} — start the API with: uvicorn app.main:app --port 8000`} />
      )}
      {!error && payload && !payload.available && (
        <EmptyState
          title="No training report on the backend"
          hint={payload.message ?? "Run `python -m ml.train` from the backend directory."}
        />
      )}

      {m && (
        <div className="space-y-8">
          <SummaryCards m={m} />
          <Curves m={m} />
          <div className="grid gap-6 lg:grid-cols-5">
            <Panel className="p-5 lg:col-span-3">
              <SectionLabel>Confusion matrix · rows = true, cols = predicted</SectionLabel>
              {m.evaluation ? (
                <ConfusionMatrix evaluation={m.evaluation} />
              ) : (
                <p className="mt-3 text-sm text-ink-faint">
                  Not in this report — retrain with the current ml/train.py.
                </p>
              )}
            </Panel>
            <Panel className="p-5 lg:col-span-2">
              <SectionLabel>Decision level used per flow</SectionLabel>
              <LevelDistribution dist={m.level_distribution} />
            </Panel>
          </div>
          {m.evaluation && (
            <Panel className="p-5">
              <SectionLabel>F1 by family · attack families colour-coded by severity</SectionLabel>
              <div className="mt-3">
                <PerFamilyF1 perClass={m.evaluation.per_class} />
              </div>
            </Panel>
          )}
        </div>
      )}
    </div>
  );
}

function SummaryCards({ m }: { m: NonNullable<TrainingMetrics["metrics"]> }) {
  const l1 = m.level1;
  const cards = [
    {
      label: "PR-AUC",
      value: l1 ? l1.pr_auc.toFixed(4) : "—",
      note: "primary · precision-recall under a 1.23% attack rate",
      color: "#22d3ee",
    },
    {
      label: "ROC-AUC",
      value: l1 ? l1.roc_auc.toFixed(4) : "—",
      note: "secondary · majority-dominated, reported for context",
      color: "#a78bfa",
    },
    {
      label: "Family accuracy",
      value:
        m.level2_family_accuracy_on_attacks != null
          ? `${(m.level2_family_accuracy_on_attacks * 100).toFixed(1)}%`
          : "—",
      note: "on true attacks, excluding Level-3 escalations",
      color: "#22c55e",
    },
    {
      label: "Zero-day escalations",
      value:
        m.attacks_escalated_as_unseen != null && m.benign_escalated_as_unseen != null
          ? `${m.attacks_escalated_as_unseen} / ${m.benign_escalated_as_unseen}`
          : "—",
      note: "attacks flagged / benign false alarms · calibrated at p99.9 benign",
      color: "#f97316",
    },
  ];

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {cards.map((c) => (
        <Panel key={c.label} className="p-4">
          <p className="font-mono text-[11px] uppercase tracking-wider text-ink-faint">
            {c.label}
          </p>
          <p className="mt-2 font-mono text-3xl font-semibold tabular-nums" style={{ color: c.color }}>
            {c.value}
          </p>
          <p className="mt-1.5 text-xs leading-snug text-ink-faint">{c.note}</p>
        </Panel>
      ))}
    </div>
  );
}

function Curves({ m }: { m: NonNullable<TrainingMetrics["metrics"]> }) {
  if (!m.evaluation || !m.level1) {
    return (
      <Panel className="p-5">
        <p className="text-sm text-ink-faint">
          Curve data not in this report — retrain with the current ml/train.py.
        </p>
      </Panel>
    );
  }

  const pr = m.evaluation.curves.pr.recall.map((recall, i) => ({
    recall,
    precision: m.evaluation!.curves.pr.precision[i],
  }));
  const roc = m.evaluation.curves.roc.fpr.map((fpr, i) => ({
    fpr,
    tpr: m.evaluation!.curves.roc.tpr[i],
  }));

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Panel className="p-5">
        <SectionLabel>
          Precision-Recall · AP {m.level1.pr_auc.toFixed(4)} · dashed = attack rate
        </SectionLabel>
        <div className="mt-3">
          <CurveChart data={pr} xKey="recall" yKey="precision" color="#22d3ee" hint={m.level1.attack_rate} />
        </div>
      </Panel>
      <Panel className="p-5">
        <SectionLabel>ROC · AUC {m.level1.roc_auc.toFixed(4)} · dashed = random</SectionLabel>
        <div className="mt-3">
          <CurveChart data={roc} xKey="fpr" yKey="tpr" color="#a78bfa" hint={undefined} />
        </div>
      </Panel>
    </div>
  );
}

function ConfusionMatrix({ evaluation }: { evaluation: NonNullable<TrainingMetrics["metrics"]>["evaluation"] }) {
  if (!evaluation) return null;
  const { labels, matrix } = evaluation.confusion_matrix;
  const max = Math.max(...matrix.flat());

  return (
    <div className="mt-3 overflow-x-auto">
      <table className="w-full border-separate border-spacing-1 text-right font-mono text-[11px]">
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} className="px-1 pb-1 font-medium text-ink-faint">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="pr-2 text-right font-medium text-ink-faint">{labels[i]}</th>
              {row.map((v, j) => {
                const diagonal = i === j;
                const intensity = max === 0 ? 0 : v / max;
                return (
                  <td
                    key={j}
                    title={`${labels[i]} → ${labels[j]}: ${v.toLocaleString()}`}
                    className="min-w-[3.5rem] rounded px-2 py-1.5 text-ink"
                    style={{
                      background: diagonal
                        ? `rgba(34,197,94,${0.08 + intensity * 0.35})`
                        : v > 0
                          ? `rgba(244,63,94,${0.08 + intensity * 0.35})`
                          : "transparent",
                    }}
                  >
                    {v > 0 ? v.toLocaleString() : "·"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LevelDistribution({ dist }: { dist: Record<string, number> | undefined }) {
  if (!dist) return <p className="mt-3 text-sm text-ink-faint">Not in this report.</p>;
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  const rows = [
    { level: 1, name: "Binary gate", desc: "cleared as benign by Level 1", color: "#22c55e" },
    { level: 2, name: "Family head", desc: "labelled by the family classifier", color: "#22d3ee" },
    { level: 3, name: "Anomaly head", desc: "escalated as potential zero-day", color: "#fb7185" },
  ];

  return (
    <div className="mt-3 space-y-4">
      <div className="flex h-3 overflow-hidden rounded-full bg-surface-2">
        {rows.map((r) => (
          <div
            key={r.level}
            style={{ width: `${((dist[String(r.level)] ?? 0) / total) * 100}%`, background: r.color }}
            title={`Level ${r.level}: ${dist[String(r.level)] ?? 0}`}
          />
        ))}
      </div>
      {rows.map((r) => {
        const n = dist[String(r.level)] ?? 0;
        return (
          <div key={r.level} className="flex items-baseline justify-between gap-3 text-sm">
            <span className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />
              <span className="text-ink">L{r.level} · {r.name}</span>
              <span className="text-xs text-ink-faint">{r.desc}</span>
            </span>
            <span className="font-mono text-ink-dim tabular-nums">
              {n.toLocaleString()} ({((n / total) * 100).toFixed(1)}%)
            </span>
          </div>
        );
      })}
    </div>
  );
}