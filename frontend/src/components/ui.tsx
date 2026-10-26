import Link from "next/link";
import type { ButtonHTMLAttributes, ReactNode } from "react";
import { styleFor } from "@/lib/types";

/** Glass card used across every page. */
export function Panel({
  children,
  className = "",
  hover = false,
}: {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}) {
  return (
    <div className={`panel ${hover ? "panel-hover" : ""} ${className}`}>
      {children}
    </div>
  );
}

/** Page header with the ambient glow and a gradient hairline. */
export function PageHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="aurora relative mb-8 overflow-hidden">
      <div className="grid-backdrop absolute inset-0" aria-hidden />
      <div className="relative z-10 flex flex-wrap items-end justify-between gap-4 py-8">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.25em] text-cyan-glow">
            {eyebrow}
          </p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">
            {title}
          </h1>
          {subtitle && (
            <p className="mt-2 max-w-2xl text-sm text-ink-dim">{subtitle}</p>
          )}
        </div>
        {actions && <div className="flex gap-2">{actions}</div>}
      </div>
      <div className="hairline relative z-10" />
    </header>
  );
}

/** Small uppercase label for metric groups. */
export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">
      {children}
    </p>
  );
}

/**
 * Verdict badge.  Colour comes from a single shared ramp so the same family is
 * always the same colour on every page, chart and alert card.
 */
export function VerdictBadge({
  label,
  level,
  size = "md",
}: {
  label: string;
  level?: number;
  size?: "sm" | "md";
}) {
  const style = familyStyle(label);
  const pad = size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs";

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${pad}`}
      style={{
        color: style.color,
        borderColor: `${style.color}55`,
        background: `${style.color}14`,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: style.color }}
      />
      {style.short}
      {level !== undefined && (
        <span className="font-mono text-[10px] opacity-70">L{level}</span>
      )}
    </span>
  );
}

/** Inline status pill for connection / model state. */
export function StatusDot({
  ok,
  label,
}: {
  ok: boolean | null;
  label: string;
}) {
  const color = ok === null ? "#64748b" : ok ? "#22c55e" : "#f43f5e";
  return (
    <span className="inline-flex items-center gap-2 text-xs text-ink-dim">
      <span
        className={`h-2 w-2 rounded-full ${ok ? "pulse-alert" : ""}`}
        style={{ background: color }}
      />
      {label}
    </span>
  );
}

/** Horizontal bar for SHAP contributions; the sign gives the direction. */
export function ContributionBar({
  contribution,
  max,
  color = "#22d3ee",
}: {
  contribution: number;
  max: number;
  color?: string;
}) {
  const pct = max === 0 ? 0 : (Math.abs(contribution) / max) * 100;
  const positive = contribution >= 0;

  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
      <div
        className="absolute inset-y-0 left-0 rounded-full transition-all duration-500"
        style={{
          width: `${Math.min(pct, 100)}%`,
          background: positive ? color : "linear-gradient(90deg,#f43f5e,#fb7185)",
        }}
      />
    </div>
  );
}

/** Primary button used for form submissions. */
export function PrimaryButton({
  children,
  loading = false,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { loading?: boolean }) {
  return (
    <button
      {...rest}
      disabled={rest.disabled || loading}
      className="relative overflow-hidden rounded-lg border border-cyan-glow/40 bg-cyan-glow/10 px-4 py-2 text-sm font-medium text-cyan-glow transition hover:bg-cyan-glow/20 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {loading && <span className="sweep absolute inset-0" aria-hidden />}
      <span className="relative">{loading ? "Analysing…" : children}</span>
    </button>
  );
}

/** Subtle secondary button. */
export function GhostButton({
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...rest}
      className="rounded-lg border border-line bg-surface-2/60 px-3 py-1.5 text-xs text-ink-dim transition hover:border-cyan-glow/40 hover:text-ink"
    >
      {children}
    </button>
  );
}

/** Error banner for failed API calls. */
export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-rose/40 bg-rose/10 px-4 py-3 text-sm text-rose">
      {message}
    </div>
  );
}

/** Empty / placeholder state. */
export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-6 py-10 text-center">
      <p className="text-sm text-ink-dim">{title}</p>
      {hint && <p className="mt-1 text-xs text-ink-faint">{hint}</p>}
    </div>
  );
}

/** Site-wide navigation. */
export function Nav() {
  const links = [
    { href: "/", label: "Overview" },
    { href: "/dashboard", label: "Dashboard" },
    { href: "/detect", label: "Live Detect" },
    { href: "/prevent", label: "Prevention" },
    { href: "/batch", label: "Batch Scan" },
    { href: "/explain", label: "Explainability" },
    { href: "/architecture", label: "Architecture" },
  ];

  return (
    <nav className="sticky top-0 z-50 border-b border-line bg-canvas/85 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-3">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="grid h-8 w-8 place-items-center rounded-lg border border-cyan-glow/40 bg-cyan-glow/10 font-mono text-xs font-bold text-cyan-glow">
            HX
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-semibold text-ink">
              H-XAI-NIDS
            </span>
            <span className="block font-mono text-[10px] uppercase tracking-wider text-ink-faint">
              Explainable NIDS
            </span>
          </span>
        </Link>

        <ul className="flex flex-wrap gap-1">
          {links.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="rounded-md px-2.5 py-1.5 text-xs text-ink-dim transition hover:bg-surface-2 hover:text-ink"
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}

/** Colour + short name for a family label; delegates to the shared ramp. */
export function familyStyle(label: string): { color: string; short: string } {
  return styleFor(label);
}
