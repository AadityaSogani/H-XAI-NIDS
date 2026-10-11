/**
 * Synthetic traffic profiles shared by the detect and explain pages.
 *
 * These are illustrative starting points built from flow-statistic name
 * patterns — not captured flows — so the UI always has something to score
 * before an analyst pastes real data.
 */

export type FormState = Record<string, string>;

export function profile(
  features: string[],
  kind: "benign" | "scan" | "web",
): FormState {
  const tuned: Record<string, [number, number, number]> = {
    duration: [0.8, 0.002, 3.0],
    iat: [0.05, 0.001, 0.4],
    pkt_len: [220, 42, 900],
    length: [220, 42, 900],
    count: [12, 2, 40],
    packets: [12, 2, 40],
    bytes: [2600, 84, 36000],
  };
  const idx = kind === "benign" ? 0 : kind === "scan" ? 1 : 2;
  const row: FormState = {};
  for (const f of features) {
    const key = Object.keys(tuned).find((k) => f.toLowerCase().includes(k));
    row[f] = String(key ? tuned[key][idx] : kind === "scan" ? 0.1 : 1);
  }
  return row;
}

/** String form state → numeric payload for the API. */
export function toPayload(
  features: string[],
  form: FormState,
): { payload: Record<string, number>; invalid: string | null } {
  const payload: Record<string, number> = {};
  for (const f of features) {
    const v = Number(form[f] ?? "");
    if (!Number.isFinite(v)) return { payload, invalid: f };
    payload[f] = v;
  }
  return { payload, invalid: null };
}
