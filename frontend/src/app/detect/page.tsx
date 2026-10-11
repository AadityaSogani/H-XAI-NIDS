"use client";

/**
 * Single-flow detection with live explanation.
 *
 * The form is driven by GET /features so it always matches the deployed
 * model's exact contract — add a feature at training time and it appears here
 * with no frontend change.
 */
import { useCallback, useEffect, useState } from "react";
import {
  ContributionBar,
  EmptyState,
  ErrorBanner,
  GhostButton,
  PageHeader,
  Panel,
  PrimaryButton,
  SectionLabel,
  VerdictBadge,
} from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { profile, toPayload, type FormState } from "@/lib/presets";
import type { Prediction } from "@/lib/types";

export default function DetectPage() {
  const [features, setFeatures] = useState<string[]>([]);
  const [form, setForm] = useState<FormState>({});
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contractError, setContractError] = useState<string | null>(null);

  useEffect(() => {
    api
      .features()
      .then((contract) => {
        setFeatures(contract.feature_names);
        setForm(profile(contract.feature_names, "benign"));
      })
      .catch((err: unknown) =>
        setContractError(
          err instanceof ApiError ? err.message : "Could not load the feature contract.",
        ),
      );
  }, []);

  const submit = useCallback(async () => {
    setLoading(true);
    setError(null);
    const { payload, invalid } = toPayload(features, form);
    if (invalid) {
      setError(`"${invalid}" must be a finite number.`);
      setLoading(false);
      return;
    }
    try {
      setResult(await api.explain(payload));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }, [features, form]);

  return (
    <div>
      <PageHeader
        eyebrow="Live detection · explainable"
        title="Analyse a flow"
        subtitle="Enter one flow's engineered features. The hierarchy returns a verdict plus the TreeSHAP terms that produced it — positive contributions push toward attack."
      />

      {contractError && (
        <ErrorBanner
          message={`${contractError} — start the API with: uvicorn app.main:app --port 8000`}
        />
      )}
      {!contractError && features.length === 0 && (
        <EmptyState title="Loading feature contract…" />
      )}

      {features.length > 0 && (
        <div className="grid gap-6 xl:grid-cols-5">
          <Panel className="p-5 xl:col-span-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <SectionLabel>
                Flow features · {features.length} inputs (model contract)
              </SectionLabel>
              <div className="flex gap-2">
                <GhostButton onClick={() => setForm(profile(features, "benign"))}>
                  Benign profile
                </GhostButton>
                <GhostButton onClick={() => setForm(profile(features, "scan"))}>
                  Scan-like profile
                </GhostButton>
                <GhostButton onClick={() => setForm(profile(features, "web"))}>
                  Web-attack profile
                </GhostButton>
              </div>
            </div>

            <div className="mt-4 grid max-h-[26rem] gap-3 overflow-y-auto pr-1 sm:grid-cols-2 lg:grid-cols-3">
              {features.map((f) => (
                <label key={f} className="block">
                  <span className="block truncate font-mono text-[10px] uppercase tracking-wide text-ink-faint">
                    {f}
                  </span>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={form[f] ?? ""}
                    onChange={(e) => setForm((prev) => ({ ...prev, [f]: e.target.value }))}
                    className="mt-1 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1.5 font-mono text-xs text-ink outline-none transition focus:border-cyan-glow/50"
                  />
                </label>
              ))}
            </div>

            <div className="mt-4 flex items-center gap-3">
              <PrimaryButton onClick={submit} loading={loading}>
                Classify + explain
              </PrimaryButton>
              {result && (
                <GhostButton onClick={() => setResult(null)}>Clear result</GhostButton>
              )}
            </div>
            <p className="mt-2 text-[11px] text-ink-faint">
              Profile buttons fill synthetic values to demo the flow — paste real
              feature values for production scoring.
            </p>
          </Panel>

          <div className="space-y-6 xl:col-span-2">
            {error && <ErrorBanner message={error} />}
            {!result && !error && (
              <Panel className="p-5">
                <EmptyState
                  title="No verdict yet"
                  hint="Pick a profile or type values, then classify."
                />
              </Panel>
            )}
            {result && <VerdictCard result={result} />}
          </div>
        </div>
      )}
    </div>
  );
}

/** Verdict summary plus the TreeSHAP waterfall and surrogate cross-check. */
function VerdictCard({ result }: { result: Prediction }) {
  const v = result.verdict;
  const shap = result.explanation?.shap ?? [];
  const maxAbs = Math.max(...shap.map((s) => Math.abs(s.contribution)), 1e-9);

  return (
    <>
      <Panel className="p-5">
        <SectionLabel>Verdict</SectionLabel>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <VerdictBadge label={v.label} level={v.level} />
          <span className="font-mono text-2xl font-semibold text-ink">
            {(v.confidence * 100).toFixed(1)}%
          </span>
          <span className="text-xs text-ink-faint">confidence</span>
        </div>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <div>
            <dt className="text-xs text-ink-faint">P(attack) · Level 1</dt>
            <dd className="font-mono text-ink">{v.p_attack.toFixed(4)}</dd>
          </div>
          <div>
            <dt className="text-xs text-ink-faint">Anomaly score · Level 3</dt>
            <dd className="font-mono text-ink">{v.anomaly_score.toFixed(4)}</dd>
          </div>
        </dl>
        {v.unseen && (
          <p className="mt-3 rounded-md border border-rose/40 bg-rose/10 px-3 py-2 text-xs leading-relaxed text-rose">
            Escalated as a potential zero-day: off the training manifold and/or
            vetoing an over-confident family call. Manual triage recommended.
          </p>
        )}
        {result.model_version && (
          <p className="mt-3 font-mono text-[10px] text-ink-faint">
            model v{result.model_version}
          </p>
        )}
      </Panel>

      <Panel className="p-5">
        <SectionLabel>SHAP waterfall · push toward attack (+) / benign (−)</SectionLabel>
        {shap.length === 0 ? (
          <p className="mt-3 text-sm text-ink-faint">No explanation returned.</p>
        ) : (
          <ul className="mt-3 space-y-2.5">
            {shap.map((s) => (
              <li key={s.feature}>
                <div className="flex items-baseline justify-between gap-2 font-mono text-[11px]">
                  <span className="truncate text-ink-dim">{s.feature}</span>
                  <span className="shrink-0 text-ink-faint">
                    = {s.value.toFixed(3)}
                    <span
                      className="ml-2"
                      style={{ color: s.contribution >= 0 ? "#22d3ee" : "#f43f5e" }}
                    >
                      {s.contribution >= 0 ? "+" : ""}
                      {s.contribution.toFixed(4)}
                    </span>
                  </span>
                </div>
                <div className="mt-1">
                  <ContributionBar contribution={s.contribution} max={maxAbs} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      {result.explanation && (
        <Panel className="p-5">
          <SectionLabel>Surrogate cross-check</SectionLabel>
          <p className="mt-2 text-sm text-ink-dim">
            Local fidelity R²:{" "}
            <span className="font-mono text-ink">
              {result.explanation.surrogate_fidelity_r2 != null
                ? result.explanation.surrogate_fidelity_r2.toFixed(3)
                : "—"}
            </span>
            {result.explanation.surrogate_fidelity_r2 != null &&
              result.explanation.surrogate_fidelity_r2 < 0.5 && (
                <span className="ml-2 text-xs text-amber">
                  low — this flow sits where the linear surrogate fits poorly;
                  trust the SHAP terms instead.
                </span>
              )}
          </p>
          {(result.explanation.surrogate?.length ?? 0) > 0 && (
            <ul className="mt-3 space-y-1 font-mono text-[11px] text-ink-dim">
              {result.explanation.surrogate!.map((s) => (
                <li key={s.feature} className="flex justify-between gap-2">
                  <span className="truncate">{s.feature}</span>
                  <span style={{ color: s.contribution >= 0 ? "#22d3ee" : "#f43f5e" }}>
                    {s.contribution >= 0 ? "+" : ""}
                    {s.contribution.toFixed(4)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      )}
    </>
  );
}


