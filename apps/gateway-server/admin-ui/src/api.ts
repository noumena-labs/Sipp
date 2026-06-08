export interface AdminSession {
  readonly authenticated: boolean;
  readonly csrfToken: string;
  readonly csrfHeader: string;
  readonly basePath: string;
}

export interface LoggedOutSession {
  readonly authenticated: false;
}

export interface DashboardOverview {
  readonly uptimeSeconds: number;
  readonly maxRequestBytes: number;
  readonly configuredConcurrencyLimit: number | null;
  readonly metrics: MetricsSnapshot;
  readonly controls: {
    readonly concurrency: ConcurrencySnapshot;
  };
  readonly security: SecuritySnapshot;
  readonly backends: unknown;
}

export interface MetricsSnapshot {
  readonly totals: TotalsSnapshot;
  readonly operations: readonly OperationSnapshot[];
  readonly targets: readonly TargetMetricsSnapshot[];
  readonly clients: readonly ClientSnapshot[];
  readonly timeseries: readonly TimeBucketSnapshot[];
  readonly recent: readonly RecentRequest[];
}

export interface TotalsSnapshot {
  readonly requests: number;
  readonly errors: number;
  readonly activeRequests: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly rateLimitHits: number;
  readonly blocklistHits: number;
  readonly p50LatencyMs: number;
  readonly p90LatencyMs: number;
  readonly p99LatencyMs: number;
}

export interface OperationSnapshot {
  readonly operation: string;
  readonly requests: number;
  readonly errors: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly averageLatencyMs: number;
}

export interface TargetMetricsSnapshot {
  readonly name: string;
  readonly requests: number;
  readonly errors: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly averageLatencyMs: number;
}

export interface ClientSnapshot {
  readonly client: string;
  readonly caller: string | null;
  readonly requests: number;
  readonly errors: number;
  readonly rateLimitHits: number;
  readonly blocklistHits: number;
  readonly totalTokens: number;
  readonly lastSeenUnixSeconds: number;
  readonly averageLatencyMs: number;
}

export interface TimeBucketSnapshot {
  readonly unixSeconds: number;
  readonly requests: number;
  readonly errors: number;
  readonly rateLimited: number;
  readonly blocked: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalTokens: number;
  readonly averageLatencyMs: number;
}

export interface RecentRequest {
  readonly unixSeconds: number;
  readonly operation: string;
  readonly requestId: string | null;
  readonly client: string;
  readonly caller: string | null;
  readonly target: string | null;
  readonly status: number;
  readonly latencyMs: number;
  readonly totalTokens: number;
  readonly streaming: boolean;
  readonly rejection: string | null;
}

export interface ConcurrencySnapshot {
  readonly limit: number | null;
  readonly active: number;
}

export interface SecuritySnapshot {
  readonly clientIpSource: string;
  readonly trustedProxyCidrs: readonly string[];
  readonly rateLimit: RateLimitSnapshot;
  readonly blocklist: readonly BlockedClientSnapshot[];
}

export interface RateLimitSnapshot {
  readonly enabled: boolean;
  readonly requestsPerMinute: number;
  readonly burst: number;
  readonly trackedClients: number;
}

export interface BlockedClientSnapshot {
  readonly client: string;
  readonly ageSeconds: number;
}

export interface RoutesResponse {
  readonly routes: readonly RouteSnapshot[];
}

export interface RouteSnapshot {
  readonly name: string;
  readonly path: string;
}

export interface TargetsResponse {
  readonly targets: readonly TargetSnapshot[];
  readonly metrics: readonly TargetMetricsSnapshot[];
}

export interface TargetSnapshot {
  readonly name: string;
  readonly kind: string;
  readonly model: string;
  readonly backend: TargetBackendSnapshot | null;
  readonly providerBaseUrl: string | null;
}

export interface TargetBackendSnapshot {
  readonly selected: string;
  readonly requested: string;
  readonly reason: string | null;
  readonly gpuOffloadExpected: boolean;
}

export interface ClientsResponse {
  readonly clients: readonly ClientSnapshot[];
  readonly recent: readonly RecentRequest[];
}

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...init,
    headers: {
      ...(init?.headers ?? {}),
      Accept: 'application/json',
    },
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Request failed with ${response.status}`);
  }
  return (await response.json()) as T;
}

export function createSession(password: string): Promise<AdminSession> {
  return fetchJson<AdminSession>('./api/session', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ password }),
  });
}

export function deleteSession(): Promise<LoggedOutSession> {
  return fetchJson<LoggedOutSession>('./api/session', {
    method: 'DELETE',
  });
}

export async function mutateJson<T>(
  path: string,
  csrfHeader: string,
  csrfToken: string,
  method: 'POST' | 'PUT' | 'DELETE',
  body?: unknown
): Promise<T> {
  return fetchJson<T>(path, {
    method,
    headers: {
      'Content-Type': 'application/json',
      [csrfHeader]: csrfToken,
    },
    body: body == null ? undefined : JSON.stringify(body),
  });
}
