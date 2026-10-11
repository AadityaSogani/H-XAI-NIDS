"use client";

/**
 * Reusable chart primitives for the dashboard.
 *
 * All charts share the severity colour ramp from lib/types so a family keeps
 * the same colour everywhere.  Recharts needs the DOM, hence "use client".
 */
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Evaluation } from "@/lib/types";
import { styleFor } from "@/lib/types";

const AXIS = { stroke: "#334155", fontSize: 11 };
const TICK = { fill: "#64748b", fontSize: 11 };

function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string }[];
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border border-line bg-surface/95 px-3 py-2 font-mono text-[11px] shadow-lg">
      <p className="text-ink-faint">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color ?? "#e6edf7" }}>
          {entry.name}: {typeof entry.value === "number" ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
}

/** PR or ROC curve.  `hint` is the baseline (0.0123 for PR under this corpus). */
export function CurveChart({
  data,
  xKey,
  yKey,
  color,
  hint,
}: {
  data: Record<string, number>[];
  xKey: string;
  yKey: string;
  color: string;
  hint?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
        <XAxis dataKey={xKey} type="number" domain={[0, 1]} tick={TICK} axisLine={AXIS} />
        <YAxis domain={[0, 1]} tick={TICK} axisLine={AXIS} />
        <Tooltip content={<ChartTooltip />} />
        {hint !== undefined && (
          <ReferenceLine y={hint} stroke="#475569" strokeDasharray="4 4" />
        )}
        <Line
          type="monotone"
          dataKey={yKey}
          stroke={color}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Horizontal F1 bars per attack family, coloured by the shared ramp. */
export function PerFamilyF1({ perClass }: { perClass: Evaluation["per_class"] }) {
  const data = Object.entries(perClass)
    .map(([name, m]) => ({
      key: name,
      name: styleFor(name).short,
      f1: m.f1,
      support: m.support,
    }))
    .sort((a, b) => b.f1 - a.f1);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" domain={[0, 1]} tick={TICK} axisLine={AXIS} />
        <YAxis type="category" dataKey="name" width={92} tick={TICK} axisLine={AXIS} />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "#1e293b55" }} />
        <Bar dataKey="f1" name="F1" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((entry) => (
            <Cell key={entry.key} fill={styleFor(entry.key).color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
