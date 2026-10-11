"use client";

/**
 * Explainability showcase: why TreeSHAP globally, a local surrogate per
 * alert, and why neither is LIME.  The live demo runs the real explainer
 * against the deployed model on a synthetic scan-like profile.
 */
import { useCallback, useState } from "react";
import {
  ContributionBar,
  ErrorBanner,
  PageHeader,
  Panel,
  PrimaryButton,
  SectionLabel,
  VerdictBadge,
} from "@/components/ui";
import { api } from "@/lib/api";
import { profile, toPayload } from "@/lib/presets";
import type { Prediction } from "@/lib/types";

const PRINCIPLES = [
  {
    title: "Global: TreeSHAP",
    body: "Exact Shapley values for tree ensembles — not approximations. Mean |SHAP| over a background sample ranks every feature by how much it moves the attack probability, in both directions. This is the model's true global structure, not a probe.",
  },
  {
    title: "Local: linear surrogate",
    body: "Each alert fits a weighted linear model on neighbourhood samples around the flow; its coefficients and fidelity R² ship with the verdict. A low R² means the flow sits in a region the surrogate cannot summarise — read the SHAP terms, not the surrogate.",
  },
  {
    title: "Why not LIME?",
    body: "LIME refits a new local model per explanation with random perturbations, so two runs on the same flow differ. That instability erodes analyst trust in an alert queue. TreeSHAP is deterministic given the background set; the surrogate reports its own fit quality instead of hiding it.",
  },
  {
    title: "Calibrated, not raw",
    body: "Explanations attach to the deployed hierarchy: SHAP terms explain the Level-1 gate probability, and the verdict tells you which level actually decided. An anomaly-head escalation is absence-of-manifold evidence — no single feature drove it.",
  },
];

export default function ExplainPage() {
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const contract = await api.features();
      const { payload, invalid } = toPayload(
        contract.feature_names,
        profile(contract.feature_names, "scan"),
      );
      if (invalid) throw new Error(`Bad contract: "${invalid}" not numeric.`);
      setResult(await api.explain(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Explanation failed.");
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div>
      <PageHeader
        eyebrow="Dual XAI · trust layer"
        title="Why the model decided that"
        subtitle="Two complementary explainers: exact TreeSHAP for global structure and per-alert attributions, plus a local surrogate that audits itself with a fidelity score."
      />

      <div className="grid gap-6 lg:grid-cols-2">
        {PRINCIPLES.map((p) => (
          <Panel key={p.title} className="p-5" hover>
            <h3 className="text-sm font-semibold text-ink">{p.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-dim">{p.body}</p>
          </Panel>
        ))}
      </div>

      <Panel className="mt-8 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <SectionLabel>Live demo · real explainer, synthetic scan-like flow</SectionLabel>
          <PrimaryButton onClick={run} loading={loading}>
            Run explanation
          </PrimaryButton>
        </div>
        {error && <div className="mt-4"><ErrorBanner message={error} /></div>}
        {result && <DemoResult result={result} />}
      </Panel>
    </div>
  );
}

function DemoResult({ result }: { result: Prediction }) {
  const v = result.verdict;
  const shap = result.explanation?.shap ?? [];
  const surrogate = result.explanation?.surrogate ?? [];
  const r2 = result.explanation?.surrogate_fidelity_r2;
  const maxAbs = Math.max(
    ...shap.map((s) => Math.abs(s.contribution)),
    ...surrogate.map((s) => Math.abs(s.contribution)),
    1e-9,
  );

  return (
    <div className="mt-5 space-y-5">
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <VerdictBadge label={v.label} level={v.level} />
        <span className="text-ink-dim">
          P(attack) <span className="font-mono text-ink">{v.p_attack.toFixed(4)}</span>
          {" · "}anomaly <span className="font-mono text-ink">{v.anomaly_score.toFixed(4)}</span>
        </span>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <SectionLabel>TreeSHAP · top terms for this flow</SectionLabel>
          <ul className="mt-3 space-y-2">
            {shap.map((s) => (
              <li key={s.feature}>
                <div className="flex items-baseline justify-between gap-2 font-mono text-[11px]">
                  <span className="truncate text-ink-dim">{s.feature}</span>
                  <span style={{ color: s.contribution >= 0 ? "#22d3ee" : "#f43f5e" }}>
                    {s.contribution >= 0 ? "+" : ""}
                    {s.contribution.toFixed(4)}
                  </span>
                </div>
                <div className="mt-1">
                  <ContributionBar contribution={s.contribution} max={maxAbs} />
                </div>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <SectionLabel>
            Local surrogate · fidelity R²{" "}
            <span style={{ color: r2 != null && r2 < 0.5 ? "#eab308" : "#22c55e" }}>
              {r2 != null ? r2.toFixed(3) : "—"}
            </span>
          </SectionLabel>
          {surrogate.length === 0 ? (
            <p className="mt-3 text-sm text-ink-faint">
              Surrogate unavailable for this flow.
            </p>
          ) : (
            <ul className="mt-3 space-y-2">
              {surrogate.map((s) => (
                <li key={s.feature}>
                  <div className="flex items-baseline justify-between gap-2 font-mono text-[11px]">
                    <span className="truncate text-ink-dim">{s.feature}</span>
                    <span style={{ color: s.contribution >= 0 ? "#22d3ee" : "#f43f5e" }}>
                      {s.contribution >= 0 ? "+" : ""}
                      {s.contribution.toFixed(4)}
                    </span>
                  </div>
                  <div className="mt-1">
                    <ContributionBar contribution={s.contribution} max={maxAbs} color="#a78bfa" />
                  </div>
                </li>
              ))}
            </ul>
          )}
          <p className="mt-3 text-[11px] leading-relaxed text-ink-faint">
            Same flow, two lenses: SHAP decomposes the ensemble&apos;s actual
            output; the surrogate approximates it locally and admits how well.
            Agreement between them raises confidence in the alert.
          </p>
        </div>
      </div>
    </div>
  );
}

