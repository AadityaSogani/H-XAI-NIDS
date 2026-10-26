"use client";

/**
 * Prevention console — the IPS half of the system.
 *
 * Detection (Live Detect) answers "is this hostile?"; this page answers the
 * operational follow-up: "what do we do about it?".  It reuses the exact same
 * feature contract (GET /features) so the form always matches the deployed
 * model, adds optional IP/port context used only to key a blocklist entry, and
 * renders the countermeasure the policy engine chose.
 *
 * Two modes mirror the backend: `prevent` enforces into a live TTL blocklist,
 * `monitor` is a dry run that reports the would-be action without disrupting
 * anything — the safe way to tune the policy first.
 */
import { useCallback, useEffect, useState, type ReactNode } from "react";
import {
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
import { actionStyle } from "@/lib/types";
import { profile, toPayload, type FormState } from "@/lib/presets";
import type {
  BlocklistResult,
  FlowContext,
  PreventionDecision,
} from "@/lib/types";

type Mode = "prevent" | "monitor";

export default function PreventPage() {
  const [features, setFeatures] = useState<string[]>([]);
  const [form, setForm] = useState<FormState>({});
  const [srcIp, setSrcIp] = useState("");
  const [dstIp, setDstIp] = useState("");
  const [dstPort, setDstPort] = useState("");
  const [protocol, setProtocol] = useState("");
  const [mode, setMode] = useState<Mode>("prevent");

  const [result, setResult] = useState<PreventionDecision | null>(null);
  const [blocklist, setBlocklist] = useState<BlocklistResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contractError, setContractError] = useState<string | null>(null);

  const refreshBlocklist = useCallback(() => {
    api
      .blocklist()
      .then(setBlocklist)
      .catch(() => setBlocklist(null));
  }, []);

  useEffect(() => {
    api
      .features()
      .then((contract) => {
        setFeatures(contract.feature_names);
        setForm(profile(contract.feature_names, "benign"));
      })
      .catch((err: unknown) =>
        setContractError(
          err instanceof ApiError
            ? err.message
            : "Could not load the feature contract.",
        ),
      );
    refreshBlocklist();
  }, [refreshBlocklist]);

  const submit = useCallback(async () => {
    setLoading(true);
    setError(null);
    const { payload, invalid } = toPayload(features, form);
    if (invalid) {
      setError(`"${invalid}" must be a finite number.`);
      setLoading(false);
      return;
    }
    const port = dstPort.trim() === "" ? null : Number(dstPort);
    if (port != null && (!Number.isInteger(port) || port < 0 || port > 65535)) {
      setError("Destination port must be an integer between 0 and 65535.");
      setLoading(false);
      return;
    }
    const ctx: FlowContext = {
      src_ip: srcIp.trim() || null,
      dst_ip: dstIp.trim() || null,
      dst_port: port,
      protocol: protocol.trim() || null,
    };
    try {
      setResult(await api.prevent(payload, ctx, mode));
      refreshBlocklist();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prevention failed.");
    } finally {
      setLoading(false);
    }
  }, [features, form, srcIp, dstIp, dstPort, protocol, mode, refreshBlocklist]);

  const remove = useCallback(
    async (key: string) => {
      try {
        await api.unblock(key);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Could not remove block.");
      } finally {
        refreshBlocklist();
      }
    },
    [refreshBlocklist],
  );

  return (
    <div>
      <PageHeader
        eyebrow="Intrusion prevention · simulated IPS"
        title="Prevention console"
        subtitle="The same three-level detector now returns a countermeasure, not just a verdict. Add optional source IP / port to key a live TTL blocklist. Monitor mode reports the action without enforcing it."
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
                  Benign
                </GhostButton>
                <GhostButton onClick={() => setForm(profile(features, "scan"))}>
                  Scan-like
                </GhostButton>
                <GhostButton onClick={() => setForm(profile(features, "web"))}>
                  Web-attack
                </GhostButton>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
              {features.map((f) => (
                <label key={f} className="block">
                  <span className="block truncate font-mono text-[10px] text-ink-faint" title={f}>
                    {f}
                  </span>
                  <input
                    value={form[f] ?? ""}
                    onChange={(e) => setForm({ ...form, [f]: e.target.value })}
                    inputMode="decimal"
                    className="mt-0.5 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-xs text-ink outline-none transition focus:border-cyan-glow/50"
                  />
                </label>
              ))}
            </div>
          </Panel>

          <div className="space-y-6 xl:col-span-2">
            <Panel className="p-5">
              <SectionLabel>Flow context · keys the action (not fed to the model)</SectionLabel>
              <p className="mt-1 text-[11px] leading-relaxed text-ink-faint">
                Source IP is preferred; a destination port is a coarse fallback. With
                neither, the decision is advisory only.
              </p>
              <div className="mt-3 grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="block font-mono text-[10px] text-ink-faint">Source IP</span>
                  <input
                    value={srcIp}
                    onChange={(e) => setSrcIp(e.target.value)}
                    placeholder="203.0.113.7"
                    className="mt-0.5 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-xs text-ink outline-none focus:border-cyan-glow/50"
                  />
                </label>
                <label className="block">
                  <span className="block font-mono text-[10px] text-ink-faint">Dest IP</span>
                  <input
                    value={dstIp}
                    onChange={(e) => setDstIp(e.target.value)}
                    placeholder="10.0.0.5"
                    className="mt-0.5 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-xs text-ink outline-none focus:border-cyan-glow/50"
                  />
                </label>
                <label className="block">
                  <span className="block font-mono text-[10px] text-ink-faint">Dest port</span>
                  <input
                    value={dstPort}
                    onChange={(e) => setDstPort(e.target.value)}
                    placeholder="443"
                    inputMode="numeric"
                    className="mt-0.5 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-xs text-ink outline-none focus:border-cyan-glow/50"
                  />
                </label>
                <label className="block">
                  <span className="block font-mono text-[10px] text-ink-faint">Protocol</span>
                  <input
                    value={protocol}
                    onChange={(e) => setProtocol(e.target.value)}
                    placeholder="tcp"
                    className="mt-0.5 w-full rounded-md border border-line bg-surface-2/60 px-2 py-1 font-mono text-xs text-ink outline-none focus:border-cyan-glow/50"
                  />
                </label>
              </div>

              <div className="mt-4">
                <SectionLabel>Mode</SectionLabel>
                <div className="mt-2 flex gap-2">
                  <ModeToggle active={mode === "prevent"} onClick={() => setMode("prevent")}>
                    Prevent · enforce
                  </ModeToggle>
                  <ModeToggle active={mode === "monitor"} onClick={() => setMode("monitor")}>
                    Monitor · dry run
                  </ModeToggle>
                </div>
              </div>

              <div className="mt-4">
                <PrimaryButton onClick={submit} loading={loading}>
                  {mode === "prevent" ? "Detect & prevent" : "Detect (monitor)"}
                </PrimaryButton>
              </div>
            </Panel>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6">
          <ErrorBanner message={error} />
        </div>
      )}

      <div className="mt-6">
        <ActionPanel result={result} />
      </div>

      <div className="mt-6">
        <BlocklistPanel blocklist={blocklist} onRemove={remove} />
      </div>
    </div>
  );
}

function ModeToggle({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-lg border px-3 py-1.5 text-xs font-medium transition ${
        active
          ? "border-cyan-glow/50 bg-cyan-glow/10 text-cyan-glow"
          : "border-line bg-surface-2/60 text-ink-dim hover:text-ink"
      }`}
    >
      {children}
    </button>
  );
}

/** Renders the detection verdict bundled with the chosen countermeasure. */
function ActionPanel({ result }: { result: PreventionDecision | null }) {
  if (!result) {
    return (
      <Panel className="p-5">
        <SectionLabel>Prevention decision</SectionLabel>
        <p className="mt-3 text-sm text-ink-faint">
          Score a flow to see the verdict and the countermeasure the policy engine
          selects for it.
        </p>
      </Panel>
    );
  }

  const { verdict, prevention } = result;
  const style = actionStyle(prevention.action);

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionLabel>Prevention decision</SectionLabel>
        <div className="flex items-center gap-2">
          <VerdictBadge label={verdict.label} level={verdict.level} />
          <span className="text-ink-faint">→</span>
          <span
            className="inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-semibold"
            style={{
              color: style.color,
              borderColor: `${style.color}55`,
              background: `${style.color}14`,
            }}
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: style.color }} />
            {style.short}
          </span>
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-3">
        <Stat label="Severity" value={`${prevention.severity} / 5`} color={style.color} />
        <Stat
          label="Enforced"
          value={prevention.enforced ? "yes · blocked" : "no"}
          color={prevention.enforced ? "#f43f5e" : "#64748b"}
        />
        <Stat
          label="TTL"
          value={prevention.ttl_seconds ? `${prevention.ttl_seconds}s` : "—"}
        />
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <Metric label="Confidence" value={verdict.confidence.toFixed(4)} />
        <Metric label="P(attack)" value={verdict.p_attack.toFixed(4)} />
        <Metric label="Anomaly" value={verdict.anomaly_score.toFixed(4)} />
        <Metric label="Subject" value={prevention.subject ?? "—"} mono />
      </dl>

      <div className="mt-4 flex flex-wrap gap-2 text-[11px]">
        {prevention.gated && <Tag text="gated by confidence → alert only" color="#eab308" />}
        {prevention.escalated && <Tag text="repeat offender · escalated" color="#f43f5e" />}
        {!prevention.enforced && !prevention.gated && (
          <Tag text="advisory · nothing keyed" color="#64748b" />
        )}
      </div>

      <p className="mt-4 rounded-md border border-line bg-surface-2/50 px-3 py-2 text-xs leading-relaxed text-ink-dim">
        {prevention.rationale}
      </p>

      <div className="mt-3">
        <SectionLabel>Suggested rule (advisory — not applied to the host)</SectionLabel>
        <pre className="mt-1.5 overflow-x-auto rounded-md border border-line bg-canvas px-3 py-2 font-mono text-[11px] text-cyan-glow">
          {prevention.rule}
        </pre>
      </div>

      {result.model_version && (
        <p className="mt-3 font-mono text-[10px] text-ink-faint">model v{result.model_version}</p>
      )}
    </Panel>
  );
}

/** Live blocklist: every subject currently blocked, with a manual override. */
function BlocklistPanel({
  blocklist,
  onRemove,
}: {
  blocklist: BlocklistResult | null;
  onRemove: (key: string) => void;
}) {
  const entries = blocklist?.entries ?? [];

  return (
    <Panel className="p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionLabel>Live blocklist · {blocklist?.total ?? 0} active</SectionLabel>
        <p className="text-[11px] text-ink-faint">
          TTL-based — entries lapse on their own; a repeat hit refreshes and escalates.
        </p>
      </div>

      {entries.length === 0 ? (
        <p className="mt-3 text-sm text-ink-faint">
          Nothing blocked. Enforce a confident attack in Prevent mode to populate this.
        </p>
      ) : (
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-ink-faint">
              <tr>
                <th className="px-2 py-2 font-medium">Subject</th>
                <th className="px-2 py-2 font-medium">Action</th>
                <th className="px-2 py-2 text-right font-medium">Severity</th>
                <th className="px-2 py-2 text-right font-medium">Hits</th>
                <th className="px-2 py-2 text-right font-medium">Expires in</th>
                <th className="px-2 py-2 font-medium">Reason</th>
                <th className="px-2 py-2" />
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60">
              {entries.map((e) => {
                const style = actionStyle(e.action);
                return (
                  <tr key={e.key} className="hover:bg-surface-2/40">
                    <td className="px-2 py-1.5 font-mono text-ink">{e.key}</td>
                    <td className="px-2 py-1.5">
                      <span className="inline-flex items-center gap-1.5" style={{ color: style.color }}>
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: style.color }} />
                        {style.short}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                      {e.severity}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                      {e.hits}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono text-ink-dim tabular-nums">
                      {Math.round(e.ttl_remaining)}s
                    </td>
                    <td className="max-w-[16rem] truncate px-2 py-1.5 text-ink-faint" title={e.reason}>
                      {e.reason}
                    </td>
                    <td className="px-2 py-1.5 text-right">
                      <button
                        onClick={() => onRemove(e.key)}
                        className="rounded-md border border-line bg-surface-2/60 px-2 py-1 text-[11px] text-ink-dim transition hover:border-rose/40 hover:text-rose"
                      >
                        Unblock
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  );
}

function Stat({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="rounded-lg border border-line bg-surface-2/40 px-3 py-2">
      <p className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">{label}</p>
      <p className="mt-1 font-mono text-lg font-semibold tabular-nums" style={color ? { color } : undefined}>
        {value}
      </p>
    </div>
  );
}

function Metric({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="text-xs text-ink-faint">{label}</dt>
      <dd className={`text-ink ${mono ? "font-mono text-xs" : "font-mono"}`}>{value}</dd>
    </div>
  );
}

function Tag({ text, color }: { text: string; color: string }) {
  return (
    <span
      className="inline-flex items-center rounded-full border px-2 py-0.5"
      style={{ color, borderColor: `${color}55`, background: `${color}14` }}
    >
      {text}
    </span>
  );
}


