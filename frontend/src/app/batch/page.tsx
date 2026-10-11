"use client";

/**
 * Batch scanning: paste rows (JSON or CSV) or upload a CSV file.
 *
 * Pasted CSV is wrapped in a File and sent to /predict/upload so parsing
 * stays server-side — one parser, one contract, no drift.
 */
import { useCallback, useRef, useState } from "react";
import {
  EmptyState,
  ErrorBanner,
  PageHeader,
  Panel,
  PrimaryButton,
  SectionLabel,
} from "@/components/ui";
import { api } from "@/lib/api";
import type { BatchResult } from "@/lib/types";
import { styleFor } from "@/lib/types";

const SAMPLE_ROWS = JSON.stringify(
  [
    { flow_duration: 0.02, fwd_pkt_len_mean: 40, flow_iat_mean: 0.001, tot_fwd_pkts: 2 },
    { flow_duration: 1.4, fwd_pkt_len_mean: 780, flow_iat_mean: 0.08, tot_fwd_pkts: 22 },
  ],
  null,
  2,
);

export default function BatchPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<BatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const run = useCallback(async (action: () => Promise<BatchResult>) => {
    setLoading(true);
    setError(null);
    try {
      setResult(await action());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Batch scan failed.");
    } finally {
      setLoading(false);
    }
  }, []);

  const runText = useCallback(
    () =>
      run(async () => {
        const trimmed = text.trim();
        if (!trimmed) throw new Error("Paste rows or choose a CSV file first.");
        if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
          const parsed = JSON.parse(trimmed) as unknown;
          const rows = Array.isArray(parsed) ? parsed : [parsed];
          return api.batch(rows as Record<string, number>[]);
        }
        return api.upload(new File([trimmed], "pasted.csv", { type: "text/csv" }));
      }).catch((err) => setError(err instanceof Error ? err.message : "Failed.")),
    [run, text],
  );

  return (
    <div>
      <PageHeader
        eyebrow="Batch scanning"
        title="Score many flows at once"
        subtitle="Paste a JSON array of feature objects, paste raw CSV, or upload a CSV export. Summary first; verdicts below, capped at 200 rows for rendering."
      />

      <div className="grid gap-6 lg:grid-cols-5">
        <Panel className="p-5 lg:col-span-3">
          <SectionLabel>Input · JSON array or CSV</SectionLabel>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={12}
            spellCheck={false}
            placeholder={'[{"flow_duration": 0.02, "tot_fwd_pkts": 2, …}]'}
            className="mt-3 w-full rounded-lg border border-line bg-surface-2/60 p-3 font-mono text-xs text-ink outline-none transition focus:border-cyan-glow/50"
          />
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <PrimaryButton onClick={runText} loading={loading}>
              Scan pasted rows
            </PrimaryButton>
            <button
              onClick={() => fileRef.current?.click()}
              className="rounded-lg border border-line bg-surface-2/60 px-3 py-1.5 text-xs text-ink-dim transition hover:border-cyan-glow/40 hover:text-ink"
            >
              Upload CSV…
            </button>
            <button
              onClick={() => setText(SAMPLE_ROWS)}
              className="rounded-lg border border-line bg-surface-2/60 px-3 py-1.5 text-xs text-ink-dim transition hover:border-cyan-glow/40 hover:text-ink"
            >
              Load sample
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".csv,text/csv"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) run(() => api.upload(file));
                e.target.value = "";
              }}
            />
          </div>
          <p className="mt-2 text-[11px] text-ink-faint">
            The model validates each row against its 30-feature contract; unknown
            columns are ignored, missing required features are rejected with a
            readable 400.
          </p>
        </Panel>

        <div className="space-y-6 lg:col-span-2">
          {error && <ErrorBanner message={error} />}
          {!result && !error && (
            <Panel className="p-5">
              <EmptyState
                title="No batch scanned yet"
                hint="Paste rows or upload a CSV to see the threat summary."
              />
            </Panel>
          )}
          {result && <BatchSummary result={result} />}
        </div>
      </div>

      {result && result.verdicts.length > 0 && <VerdictTable result={result} />}
    </div>
  );
}

function BatchSummary({ result }: { result: BatchResult }) {
  const benign = result.total - result.attacks;
  const byFamily = new Map<string, number>();
  for (const v of result.verdicts) {
    byFamily.set(v.label, (byFamily.get(v.label) ?? 0) + 1);
  }
  const families = [...byFamily.entries()].sort((a, b) => b[1] - a[1]);

  return (
    <Panel className="p-5">
      <SectionLabel>Threat summary</SectionLabel>
      <div className="mt-3 grid grid-cols-3 gap-3 text-center">
        {[
          { label: "Total", value: result.total, color: "#e6edf7" },
          { label: "Attacks", value: result.attacks, color: "#f43f5e" },
          { label: "Unseen", value: result.unseen, color: "#fb7185" },
        ].map((c) => (
          <div key={c.label}>
            <p className="font-mono text-2xl font-semibold tabular-nums" style={{ color: c.color }}>
              {c.value.toLocaleString()}
            </p>
            <p className="text-[11px] uppercase tracking-wide text-ink-faint">{c.label}</p>
          </div>
        ))}
      </div>

      <div className="mt-4 flex h-2.5 overflow-hidden rounded-full bg-surface-2">
        <div
          className="bg-emerald"
          style={{ width: `${(benign / Math.max(result.total, 1)) * 100}%` }}
          title={`${benign} benign`}
        />
        <div
          className="bg-rose"
          style={{ width: `${(result.attacks / Math.max(result.total, 1)) * 100}%` }}
          title={`${result.attacks} attacks`}
        />
      </div>

      <ul className="mt-4 space-y-1.5">
        {families.map(([label, n]) => (
          <li key={label} className="flex justify-between font-mono text-xs">
            <span className="text-ink-dim">{label}</span>
            <span className="text-ink-faint tabular-nums">
              {n.toLocaleString()} ({((n / Math.max(result.total, 1)) * 100).toFixed(1)}%)
            </span>
          </li>
        ))}
      </ul>
    </Panel>
  );
}

function VerdictTable({ result }: { result: BatchResult }) {
  const shown = result.verdicts.slice(0, 200);
  return (
    <Panel className="mt-6 p-5">
      <SectionLabel>
        Verdicts · showing {shown.length} of {result.verdicts.length}
      </SectionLabel>
      <div className="mt-3 max-h-[32rem] overflow-y-auto">
        <table className="w-full text-left text-xs">
          <thead className="sticky top-0 bg-surface">
            <tr className="text-ink-faint">
              <th className="px-2 py-2 font-medium">#</th>
              <th className="px-2 py-2 font-medium">Label</th>
              <th className="px-2 py-2 font-medium">Level</th>
              <th className="px-2 py-2 text-right font-medium">Confidence</th>
              <th className="px-2 py-2 text-right font-medium">P(attack)</th>
              <th className="px-2 py-2 text-right font-medium">Anomaly</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line/60">
            {shown.map((v, i) => (
              <tr key={i} className="hover:bg-surface-2/40">
                <td className="px-2 py-1.5 font-mono text-ink-faint">{i + 1}</td>
                <td className="px-2 py-1.5">
                  <span
                    className="inline-flex items-center gap-1.5 font-medium"
                    style={{ color: familyColor(v.label) }}
                  >
                    <span
                      className="h-1.5 w-1.5 rounded-full"
                      style={{ background: familyColor(v.label) }}
                    />
                    {v.label}
                  </span>
                </td>
                <td className="px-2 py-1.5 font-mono text-ink-faint">L{v.level}</td>
                <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                  {(v.confidence * 100).toFixed(1)}%
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                  {v.p_attack.toFixed(3)}
                </td>
                <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                  {v.anomaly_score.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/** Shared colour ramp — delegated to FAMILY_STYLE in lib/types. */
function familyColor(label: string): string {
  return styleFor(label).color;
}

