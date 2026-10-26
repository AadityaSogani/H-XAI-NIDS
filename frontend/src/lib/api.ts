import type {
  BatchResult,
  BlocklistResult,
  FeatureContract,
  FlowContext,
  Health,
  Prediction,
  PreventionDecision,
  TrainingMetrics,
} from "./types";

/**
 * Thin client for the H-XAI-NIDS FastAPI backend.
 *
 * Every call goes through {@link request} so failures surface as a readable
 * message instead of a raw `fetch` rejection.  A 503 (no model loaded) is
 * deliberately distinguishable from a network error: the first means the API is
 * healthy but untrained, the second means it is unreachable — very different
 * things for an operator to act on.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly kind: "network" | "server" | "validation" = "server",
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      // The dashboard polls health; never serve a stale status.
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      `Cannot reach the detection API at ${API_BASE}. Is the backend running?`,
      0,
      "network",
    );
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      // Non-JSON error body; the status line is enough.
    }

    if (response.status === 503) {
      throw new ApiError(detail, 503, "server");
    }
    if (response.status === 400 || response.status === 422) {
      throw new ApiError(detail, response.status, "validation");
    }
    throw new ApiError(detail, response.status);
  }

  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  features: () => request<FeatureContract>("/features"),

  metrics: () => request<TrainingMetrics>("/metrics"),

  /** Classify a single flow. */
  predict: (features: Record<string, number>) =>
    request<Prediction>("/predict", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  /** Classify a flow and explain the decision with SHAP. */
  explain: (features: Record<string, number>) =>
    request<Prediction>("/predict/explain", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),

  /** Scan many flows at once. */
  batch: (rows: Record<string, number>[]) =>
    request<BatchResult>("/predict/batch", {
      method: "POST",
      body: JSON.stringify({ rows }),
    }),

  /** Score an uploaded CSV of flows. */
  upload: async (file: File): Promise<BatchResult> => {
    let response: Response;
    try {
      response = await fetch(`${API_BASE}/predict/upload`, {
        method: "POST",
        body: file,
        headers: { "Content-Type": file.type || "text/csv" },
        cache: "no-store",
      });
    } catch {
      throw new ApiError(
        `Cannot reach the detection API at ${API_BASE}.`,
        0,
        "network",
      );
    }

    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = (await response.json()) as { detail?: string };
        if (body.detail) detail = body.detail;
      } catch {
        /* keep the status line */
      }
      throw new ApiError(detail, response.status);
    }

    return (await response.json()) as BatchResult;
  },

  /**
   * Detect a flow AND return a countermeasure.  `context` is optional metadata
   * (src IP / port) used only to key the action — never fed to the model.
   * `mode: "monitor"` is a dry run that reports the action without enforcing.
   */
  prevent: (
    features: Record<string, number>,
    context: FlowContext | null,
    mode: "prevent" | "monitor" = "prevent",
  ) =>
    request<PreventionDecision>("/prevent", {
      method: "POST",
      body: JSON.stringify({ features, context, mode }),
    }),

  /** The live blocklist: subjects currently blocked, soonest-expiry first. */
  blocklist: () => request<BlocklistResult>("/prevent/blocklist"),

  /** Manually clear one block before its TTL expires. */
  unblock: (key: string) =>
    request<{ removed: string; remaining: number }>(
      `/prevent/blocklist/${encodeURIComponent(key)}`,
      { method: "DELETE" },
    ),
};
