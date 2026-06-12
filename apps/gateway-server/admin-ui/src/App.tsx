import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import type { ReactElement, ReactNode } from 'react';
import {
  Activity,
  Ban,
  Cpu,
  Gauge,
  ListTree,
  Lock,
  Network,
  RefreshCw,
  Route,
  Shield,
  SlidersHorizontal,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  type AdminSession,
  ApiError,
  type ClientsResponse,
  createSession,
  type DashboardOverview,
  deleteSession,
  fetchJson,
  mutateJson,
  type RoutesResponse,
  type TargetsResponse,
} from './api.js';
import {
  errorRate,
  formatBytes,
  formatDuration,
  formatNumber,
  sanitizeConcurrencyLimit,
  sanitizeRateLimit,
  toChartPoints,
} from './model.js';
import './styles.css';

type ViewId = 'overview' | 'traffic' | 'security' | 'targets' | 'routes';

const views: readonly { readonly id: ViewId; readonly label: string; readonly icon: typeof Activity }[] = [
  { id: 'overview', label: 'Overview', icon: Activity },
  { id: 'traffic', label: 'Traffic', icon: Network },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'targets', label: 'Targets', icon: Cpu },
  { id: 'routes', label: 'Routes', icon: Route },
];

export default function App(): ReactElement {
  const [view, setView] = useState<ViewId>('overview');
  const queryClient = useQueryClient();
  const session = useQuery({
    queryKey: ['session'],
    queryFn: () => fetchJson<AdminSession>('./api/session'),
    retry: false,
  });
  const overview = useQuery({
    queryKey: ['overview'],
    queryFn: () => fetchJson<DashboardOverview>('./api/overview'),
    refetchInterval: 5000,
    enabled: session.data != null,
  });
  const routes = useQuery({
    queryKey: ['routes'],
    queryFn: () => fetchJson<RoutesResponse>('./api/routes'),
    refetchInterval: 15000,
    enabled: session.data != null,
  });
  const targets = useQuery({
    queryKey: ['targets'],
    queryFn: () => fetchJson<TargetsResponse>('./api/targets'),
    refetchInterval: 10000,
    enabled: session.data != null,
  });
  const clients = useQuery({
    queryKey: ['clients'],
    queryFn: () => fetchJson<ClientsResponse>('./api/clients'),
    refetchInterval: 5000,
    enabled: session.data != null,
  });
  const login = useMutation({
    mutationFn: createSession,
    onSuccess: (data) => {
      queryClient.setQueryData(['session'], data);
      void queryClient.invalidateQueries({ queryKey: ['overview'] });
      void queryClient.invalidateQueries({ queryKey: ['routes'] });
      void queryClient.invalidateQueries({ queryKey: ['targets'] });
      void queryClient.invalidateQueries({ queryKey: ['clients'] });
    },
  });
  const logout = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      queryClient.removeQueries({ queryKey: ['overview'] });
      queryClient.removeQueries({ queryKey: ['routes'] });
      queryClient.removeQueries({ queryKey: ['targets'] });
      queryClient.removeQueries({ queryKey: ['clients'] });
      queryClient.setQueryData(['session'], undefined);
      void session.refetch();
    },
  });

  if (session.isLoading) {
    return <ShellNotice title="Loading Gateway Admin" detail="Opening the management dashboard." />;
  }
  if (session.isError) {
    const error = session.error;
    if (error instanceof ApiError && error.status === 401) {
      return (
        <LoginPanel
          error={login.isError ? 'Invalid password.' : null}
          onSubmit={(password) => login.mutate(password)}
          pending={login.isPending}
        />
      );
    }
    return <ShellNotice title="Dashboard Error" detail="The session endpoint is unavailable." />;
  }
  if (session.data == null || overview.isLoading) {
    return <ShellNotice title="Loading Gateway Admin" detail="Opening the management dashboard." />;
  }
  if (overview.isError || overview.data == null) {
    return <ShellNotice title="Dashboard Error" detail="The overview endpoint is unavailable." />;
  }

  const refresh = (): void => {
    void session.refetch();
    void overview.refetch();
    void routes.refetch();
    void targets.refetch();
    void clients.refetch();
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Gauge aria-hidden="true" />
          <div>
            <strong>Sipp</strong>
            <span>Gateway Admin</span>
          </div>
        </div>
        <nav className="nav-list" aria-label="Dashboard views">
          {views.map((item) => {
            const Icon = item.icon;
            return (
              <button
                className={item.id === view ? 'nav-item active' : 'nav-item'}
                key={item.id}
                onClick={() => setView(item.id)}
                type="button"
              >
                <Icon aria-hidden="true" />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="logout-form">
          <button
            className="secondary full-width"
            disabled={logout.isPending}
            onClick={() => logout.mutate()}
            type="button"
          >
            <Lock aria-hidden="true" />
            Sign out
          </button>
        </div>
      </aside>
      <main className="workspace">
        <header className="topbar">
          <div>
            <p className="eyebrow">Management listener</p>
            <h1>{views.find((item) => item.id === view)?.label}</h1>
          </div>
          <button className="icon-button" onClick={refresh} type="button" title="Refresh dashboard">
            <RefreshCw aria-hidden="true" />
          </button>
        </header>
        {view === 'overview' && <OverviewPanel data={overview.data} />}
        {view === 'traffic' &&
          (clients.data == null ? (
            <PanelNotice title="Traffic unavailable" detail="The clients endpoint is still loading or failed." />
          ) : (
            <TrafficPanel data={overview.data} clients={clients.data} />
          ))}
        {view === 'security' && <SecurityPanel data={overview.data} session={session.data} />}
        {view === 'targets' &&
          (targets.data == null ? (
            <PanelNotice title="Targets unavailable" detail="The targets endpoint is still loading or failed." />
          ) : (
            <TargetsPanel data={targets.data} />
          ))}
        {view === 'routes' &&
          (routes.data == null ? (
            <PanelNotice title="Routes unavailable" detail="The routes endpoint is still loading or failed." />
          ) : (
            <RoutesPanel routes={routes.data} overview={overview.data} />
          ))}
      </main>
    </div>
  );
}

function LoginPanel({
  error,
  onSubmit,
  pending,
}: {
  readonly error: string | null;
  readonly onSubmit: (password: string) => void;
  readonly pending: boolean;
}): ReactElement {
  const [password, setPassword] = useState('');
  return (
    <main className="auth-shell">
      <form
        className="auth-panel"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit(password);
        }}
      >
        <div className="brand auth-brand">
          <Gauge aria-hidden="true" />
          <div>
            <strong>Sipp</strong>
            <span>Gateway Admin</span>
          </div>
        </div>
        {error != null && <p className="form-error">{error}</p>}
        <label>
          Password
          <input
            autoComplete="current-password"
            autoFocus
            onChange={(event) => setPassword(event.target.value)}
            type="password"
            value={password}
          />
        </label>
        <button disabled={pending || password.length === 0} type="submit">
          <Lock aria-hidden="true" />
          Sign in
        </button>
      </form>
    </main>
  );
}

function OverviewPanel({ data }: { readonly data: DashboardOverview }): ReactElement {
  const chartPoints = useMemo(() => toChartPoints(data.metrics.timeseries), [data.metrics.timeseries]);
  return (
    <section className="panel-stack">
      <div className="metric-grid">
        <MetricCard label="Requests" value={formatNumber(data.metrics.totals.requests)} />
        <MetricCard label="Errors" value={formatNumber(data.metrics.totals.errors)} tone="warning" />
        <MetricCard label="Active" value={formatNumber(data.metrics.totals.activeRequests)} />
        <MetricCard label="Tokens" value={formatNumber(data.metrics.totals.totalTokens)} />
        <MetricCard label="P90 latency" value={`${formatNumber(data.metrics.totals.p90LatencyMs)} ms`} />
        <MetricCard label="Uptime" value={formatDuration(data.uptimeSeconds)} />
      </div>
      <div className="chart-row">
        <ChartPanel title="Request rate">
          <ResponsiveContainer height={260} width="100%">
            <AreaChart data={chartPoints}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" minTickGap={28} />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Area dataKey="requests" fill="#75aadb" stroke="#1f6feb" type="monotone" />
              <Area dataKey="errors" fill="#f2a29b" stroke="#b42318" type="monotone" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartPanel>
        <ChartPanel title="Latency">
          <ResponsiveContainer height={260} width="100%">
            <LineChart data={chartPoints}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" minTickGap={28} />
              <YAxis />
              <Tooltip />
              <Line dataKey="latency" dot={false} stroke="#257a5a" strokeWidth={2} type="monotone" />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>
      </div>
      <section className="table-panel">
        <div className="section-heading">
          <h2>Operations</h2>
          <span>Error rate by gateway operation</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Requests</th>
              <th>Errors</th>
              <th>Error rate</th>
              <th>Avg latency</th>
              <th>Tokens</th>
            </tr>
          </thead>
          <tbody>
            {data.metrics.operations.map((operation) => (
              <tr key={operation.operation}>
                <td>{operation.operation}</td>
                <td>{formatNumber(operation.requests)}</td>
                <td>{formatNumber(operation.errors)}</td>
                <td>{errorRate(operation.requests, operation.errors)}</td>
                <td>{formatNumber(operation.averageLatencyMs)} ms</td>
                <td>{formatNumber(operation.totalTokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </section>
  );
}

function TrafficPanel({
  data,
  clients,
}: {
  readonly data: DashboardOverview;
  readonly clients: ClientsResponse;
}): ReactElement {
  const chartPoints = useMemo(() => toChartPoints(data.metrics.timeseries), [data.metrics.timeseries]);
  return (
    <section className="panel-stack">
      <ChartPanel title="Traffic and denials">
        <ResponsiveContainer height={260} width="100%">
          <BarChart data={chartPoints}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" minTickGap={28} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="requests" fill="#1f6feb" />
            <Bar dataKey="rateLimited" fill="#d97706" />
            <Bar dataKey="blocked" fill="#b42318" />
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>
      <section className="table-panel">
        <div className="section-heading">
          <h2>Clients</h2>
          <span>In-memory top client summary</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Client</th>
              <th>Caller</th>
              <th>Requests</th>
              <th>Errors</th>
              <th>Denied</th>
              <th>Tokens</th>
              <th>Avg latency</th>
            </tr>
          </thead>
          <tbody>
            {clients.clients.map((client) => (
              <tr key={client.client}>
                <td>{client.client}</td>
                <td>{client.caller ?? '-'}</td>
                <td>{formatNumber(client.requests)}</td>
                <td>{formatNumber(client.errors)}</td>
                <td>{formatNumber(client.rateLimitHits + client.blocklistHits)}</td>
                <td>{formatNumber(client.totalTokens)}</td>
                <td>{formatNumber(client.averageLatencyMs)} ms</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="table-panel">
        <div className="section-heading">
          <h2>Recent events</h2>
          <span>Ephemeral request history</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Operation</th>
              <th>Status</th>
              <th>Client</th>
              <th>Target</th>
              <th>Latency</th>
              <th>Tokens</th>
            </tr>
          </thead>
          <tbody>
            {clients.recent.map((event, index) => (
              <tr key={`${event.unixSeconds}-${event.requestId ?? index}`}>
                <td>{event.operation}</td>
                <td>{event.rejection ?? event.status}</td>
                <td>{event.client}</td>
                <td>{event.target ?? '-'}</td>
                <td>{formatNumber(event.latencyMs)} ms</td>
                <td>{formatNumber(event.totalTokens)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </section>
  );
}

function SecurityPanel({
  data,
  session,
}: {
  readonly data: DashboardOverview;
  readonly session: AdminSession;
}): ReactElement {
  const queryClient = useQueryClient();
  const [enabled, setEnabled] = useState(data.security.rateLimit.enabled);
  const [requestsPerMinute, setRequestsPerMinute] = useState(
    String(data.security.rateLimit.requestsPerMinute)
  );
  const [burst, setBurst] = useState(String(data.security.rateLimit.burst));
  const [blockClient, setBlockClient] = useState('');
  const [concurrency, setConcurrency] = useState(data.controls.concurrency.limit?.toString() ?? '');
  const invalidate = (): void => {
    void queryClient.invalidateQueries({ queryKey: ['overview'] });
    void queryClient.invalidateQueries({ queryKey: ['clients'] });
  };
  const rateLimit = useMutation({
    mutationFn: () =>
      mutateJson('./api/security/rate-limit', session.csrfHeader, session.csrfToken, 'PUT', {
        enabled,
        requestsPerMinute: sanitizeRateLimit(requestsPerMinute),
        burst: sanitizeRateLimit(burst),
      }),
    onSuccess: invalidate,
  });
  const addBlock = useMutation({
    mutationFn: () =>
      mutateJson(
        `./api/security/blocklist/${encodeURIComponent(blockClient.trim())}`,
        session.csrfHeader,
        session.csrfToken,
        'POST'
      ),
    onSuccess: () => {
      setBlockClient('');
      invalidate();
    },
  });
  const removeBlock = useMutation({
    mutationFn: (client: string) =>
      mutateJson(
        `./api/security/blocklist/${encodeURIComponent(client)}`,
        session.csrfHeader,
        session.csrfToken,
        'DELETE'
      ),
    onSuccess: invalidate,
  });
  const concurrencyMutation = useMutation({
    mutationFn: () =>
      mutateJson('./api/controls/concurrency', session.csrfHeader, session.csrfToken, 'PUT', {
        limit: sanitizeConcurrencyLimit(concurrency),
      }),
    onSuccess: invalidate,
  });

  return (
    <section className="panel-stack">
      <div className="metric-grid">
        <MetricCard label="Rate-limit hits" value={formatNumber(data.metrics.totals.rateLimitHits)} />
        <MetricCard label="Blocklist hits" value={formatNumber(data.metrics.totals.blocklistHits)} />
        <MetricCard label="Tracked buckets" value={formatNumber(data.security.rateLimit.trackedClients)} />
        <MetricCard label="Blocked clients" value={formatNumber(data.security.blocklist.length)} />
      </div>
      <section className="control-grid">
        <div className="control-panel">
          <div className="section-heading">
            <h2><SlidersHorizontal aria-hidden="true" /> Rate limit</h2>
            <span>Runtime-only token bucket settings</span>
          </div>
          <label className="check-row">
            <input checked={enabled} onChange={(event) => setEnabled(event.target.checked)} type="checkbox" />
            Enabled
          </label>
          <label>
            Requests/minute
            <input value={requestsPerMinute} onChange={(event) => setRequestsPerMinute(event.target.value)} />
          </label>
          <label>
            Burst
            <input value={burst} onChange={(event) => setBurst(event.target.value)} />
          </label>
          <button onClick={() => rateLimit.mutate()} type="button">Apply rate limit</button>
        </div>
        <div className="control-panel">
          <div className="section-heading">
            <h2><Ban aria-hidden="true" /> Blocklist</h2>
            <span>Manual in-memory client blocks</span>
          </div>
          <label>
            Client IP
            <input value={blockClient} onChange={(event) => setBlockClient(event.target.value)} />
          </label>
          <button disabled={blockClient.trim().length === 0} onClick={() => addBlock.mutate()} type="button">
            Block client
          </button>
          <div className="pill-list">
            {data.security.blocklist.map((entry) => (
              <button key={entry.client} onClick={() => removeBlock.mutate(entry.client)} type="button">
                {entry.client}
              </button>
            ))}
          </div>
        </div>
        <div className="control-panel">
          <div className="section-heading">
            <h2><ListTree aria-hidden="true" /> Concurrency</h2>
            <span>Blank means unbounded after apply</span>
          </div>
          <label>
            Limit
            <input value={concurrency} onChange={(event) => setConcurrency(event.target.value)} />
          </label>
          <p className="muted">Active requests: {formatNumber(data.controls.concurrency.active)}</p>
          <button onClick={() => concurrencyMutation.mutate()} type="button">Apply concurrency</button>
        </div>
      </section>
    </section>
  );
}

function TargetsPanel({
  data,
}: {
  readonly data: TargetsResponse;
}): ReactElement {
  const metricsByTarget = new Map(data.metrics.map((target) => [target.name, target]));
  return (
    <section className="table-panel">
      <div className="section-heading">
        <h2>Targets</h2>
        <span>Configured local and provider endpoints</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Kind</th>
            <th>Model</th>
            <th>Backend</th>
            <th>Requests</th>
            <th>Tokens</th>
          </tr>
        </thead>
        <tbody>
          {data.targets.map((target) => {
            const targetMetrics = metricsByTarget.get(target.name);
            return (
              <tr key={target.name}>
                <td>{target.name}</td>
                <td>{target.kind}</td>
                <td>{target.model}</td>
                <td>{target.backend?.selected ?? target.providerBaseUrl ?? 'provider'}</td>
                <td>{formatNumber(targetMetrics?.requests ?? 0)}</td>
                <td>{formatNumber(targetMetrics?.totalTokens ?? 0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function RoutesPanel({
  routes,
  overview,
}: {
  readonly routes: RoutesResponse;
  readonly overview: DashboardOverview;
}): ReactElement {
  return (
    <section className="panel-stack">
      <div className="metric-grid">
        <MetricCard label="Max body" value={formatBytes(overview.maxRequestBytes)} />
        <MetricCard
          label="Boot concurrency"
          value={overview.configuredConcurrencyLimit == null ? 'unbounded' : String(overview.configuredConcurrencyLimit)}
        />
        <MetricCard label="Client IP source" value={overview.security.clientIpSource} />
        <MetricCard label="Trusted proxies" value={String(overview.security.trustedProxyCidrs.length)} />
      </div>
      <section className="table-panel">
        <div className="section-heading">
          <h2>Routes</h2>
          <span>Public and management paths</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Path</th>
            </tr>
          </thead>
          <tbody>
            {routes.routes.map((route) => (
              <tr key={route.name}>
                <td>{route.name}</td>
                <td>{route.path}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </section>
  );
}

function MetricCard({
  label,
  value,
  tone,
}: {
  readonly label: string;
  readonly value: string;
  readonly tone?: 'warning';
}): ReactElement {
  return (
    <div className={tone === 'warning' ? 'metric-card warning' : 'metric-card'}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ChartPanel({
  title,
  children,
}: {
  readonly title: string;
  readonly children: ReactNode;
}): ReactElement {
  return (
    <section className="chart-panel">
      <div className="section-heading">
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function ShellNotice({
  title,
  detail,
}: {
  readonly title: string;
  readonly detail: string;
}): ReactElement {
  return (
    <main className="notice-shell">
      <section>
        <h1>{title}</h1>
        <p>{detail}</p>
      </section>
    </main>
  );
}

function PanelNotice({
  title,
  detail,
}: {
  readonly title: string;
  readonly detail: string;
}): ReactElement {
  return (
    <section className="table-panel panel-notice">
      <h2>{title}</h2>
      <p>{detail}</p>
    </section>
  );
}
