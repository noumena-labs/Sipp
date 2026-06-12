use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use sipp::core::TokenUsage;
use sipp::gateway_core::Operation;
use serde::Serialize;

const TIMESERIES_BUCKET_SECONDS: u64 = 10;
const MAX_TIMESERIES_BUCKETS: usize = 180;
const MAX_RECENT_EVENTS: usize = 96;
const MAX_LATENCY_SAMPLES: usize = 512;
const MAX_CLIENTS: usize = 512;

/// Application-owned low-cardinality gateway metrics.
pub struct GatewayMetrics {
    requests: [AtomicU64; 3],
    errors: [AtomicU64; 3],
    active_requests: AtomicU64,
    state: Mutex<DashboardMetrics>,
}

/// Point-in-time metric values for one gateway operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationMetricSnapshot {
    pub operation: &'static str,
    pub requests: u64,
    pub errors: u64,
}

/// One completed request for dashboard and Prometheus accounting.
pub(crate) struct GatewayRequestRecord {
    pub(crate) operation: Operation,
    pub(crate) request_id: Option<String>,
    pub(crate) client: String,
    pub(crate) caller: Option<String>,
    pub(crate) target: Option<String>,
    pub(crate) status: StatusCode,
    pub(crate) duration: Duration,
    pub(crate) usage: Option<TokenUsage>,
    pub(crate) streaming: bool,
    pub(crate) rejection: Option<GatewayRejectionKind>,
}

/// Security rejection category recorded in dashboard telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GatewayRejectionKind {
    Blocked,
    ClientClosed,
    RateLimited,
}

impl GatewayMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self {
            requests: std::array::from_fn(|_| AtomicU64::new(0)),
            errors: std::array::from_fn(|_| AtomicU64::new(0)),
            active_requests: AtomicU64::new(0),
            state: Mutex::new(DashboardMetrics::default()),
        }
    }

    /// Record that an application request has started.
    pub(crate) fn request_started(&self, _operation: Operation, _request_id: Option<&str>) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record final request telemetry.
    pub(crate) fn request_finished(&self, record: GatewayRequestRecord) {
        let _ = self
            .active_requests
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |active| {
                Some(active.saturating_sub(1))
            });

        let index = operation_index(record.operation);
        self.requests[index].fetch_add(1, Ordering::Relaxed);
        if !record.status.is_success() {
            self.errors[index].fetch_add(1, Ordering::Relaxed);
        }
        let mut state = match self.state.lock() {
            Ok(state) => state,
            Err(error) => error.into_inner(),
        };
        state.record(record);
    }

    /// Render Prometheus text exposition.
    pub fn render(&self) -> Result<String, &'static str> {
        let mut output = String::new();
        for operation in [Operation::Query, Operation::Chat, Operation::Embed] {
            let index = operation_index(operation);
            let name = operation_name(operation);
            let _ = writeln!(
                output,
                "sipp_gateway_requests_total{{operation=\"{name}\"}} {}",
                self.requests[index].load(Ordering::Relaxed)
            );
            let _ = writeln!(
                output,
                "sipp_gateway_errors_total{{operation=\"{name}\"}} {}",
                self.errors[index].load(Ordering::Relaxed)
            );
        }
        let _ = writeln!(
            output,
            "sipp_gateway_active_requests {}",
            self.active_requests.load(Ordering::Relaxed)
        );
        let state = self
            .state
            .lock()
            .map_err(|_| "dashboard metrics are unavailable")?;
        let _ = writeln!(
            output,
            "sipp_gateway_rate_limit_hits_total {}",
            state.rate_limit_hits
        );
        let _ = writeln!(
            output,
            "sipp_gateway_blocklist_hits_total {}",
            state.blocklist_hits
        );
        let _ = writeln!(
            output,
            "sipp_gateway_input_tokens_total {}",
            state.input_tokens
        );
        let _ = writeln!(
            output,
            "sipp_gateway_output_tokens_total {}",
            state.output_tokens
        );
        for (class, count) in &state.status_classes {
            let _ = writeln!(
                output,
                "sipp_gateway_responses_total{{status_class=\"{class}\"}} {count}"
            );
        }
        Ok(output)
    }

    /// Return low-cardinality metric counters for dashboard rendering.
    pub fn snapshot(&self) -> [OperationMetricSnapshot; 3] {
        [Operation::Query, Operation::Chat, Operation::Embed].map(|operation| {
            let index = operation_index(operation);
            OperationMetricSnapshot {
                operation: operation_name(operation),
                requests: self.requests[index].load(Ordering::Relaxed),
                errors: self.errors[index].load(Ordering::Relaxed),
            }
        })
    }

    pub(crate) fn dashboard_snapshot(&self) -> Result<GatewayDashboardSnapshot, &'static str> {
        let counters = self.snapshot();
        let active_requests = self.active_requests.load(Ordering::Relaxed);
        self.state
            .lock()
            .map(|state| state.snapshot(counters, active_requests))
            .map_err(|_| "dashboard metrics are unavailable")
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Default)]
struct DashboardMetrics {
    total_requests: u64,
    total_errors: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    rate_limit_hits: u64,
    blocklist_hits: u64,
    status_classes: BTreeMap<&'static str, u64>,
    operations: BTreeMap<&'static str, OperationSummary>,
    clients: BTreeMap<String, ClientSummary>,
    targets: BTreeMap<String, TargetSummary>,
    timeseries: VecDeque<TimeBucket>,
    recent: VecDeque<RecentRequest>,
    latency_samples: VecDeque<u64>,
}

impl DashboardMetrics {
    fn record(&mut self, record: GatewayRequestRecord) {
        self.total_requests = self.total_requests.saturating_add(1);
        if !record.status.is_success() {
            self.total_errors = self.total_errors.saturating_add(1);
        }
        match record.rejection {
            Some(GatewayRejectionKind::Blocked) => {
                self.blocklist_hits = self.blocklist_hits.saturating_add(1)
            }
            Some(GatewayRejectionKind::ClientClosed) => {}
            Some(GatewayRejectionKind::RateLimited) => {
                self.rate_limit_hits = self.rate_limit_hits.saturating_add(1)
            }
            None => {}
        }

        let input_tokens = u64::from(
            record
                .usage
                .and_then(|usage| usage.input_tokens)
                .unwrap_or(0),
        );
        let output_tokens = u64::from(
            record
                .usage
                .and_then(|usage| usage.output_tokens)
                .unwrap_or(0),
        );
        let total_tokens = u64::from(
            record
                .usage
                .and_then(|usage| usage.total_tokens)
                .unwrap_or_else(|| input_tokens.saturating_add(output_tokens) as u32),
        );
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(total_tokens);

        let status_class = status_class(record.status);
        *self.status_classes.entry(status_class).or_default() += 1;
        let latency_ms = millis(record.duration);
        push_bounded(&mut self.latency_samples, latency_ms, MAX_LATENCY_SAMPLES);

        self.operations
            .entry(operation_name(record.operation))
            .or_default()
            .record(
                &record,
                input_tokens,
                output_tokens,
                total_tokens,
                latency_ms,
            );
        if let Some(target) = record.target.as_deref() {
            self.targets.entry(target.to_string()).or_default().record(
                &record,
                input_tokens,
                output_tokens,
                total_tokens,
                latency_ms,
            );
        }

        let client = self.clients.entry(record.client.clone()).or_default();
        client.record(
            &record,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
        );
        prune_clients(&mut self.clients);

        self.record_bucket(
            &record,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
        );
        push_bounded(
            &mut self.recent,
            RecentRequest::from_record(&record, latency_ms, total_tokens),
            MAX_RECENT_EVENTS,
        );
    }

    fn record_bucket(
        &mut self,
        record: &GatewayRequestRecord,
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        latency_ms: u64,
    ) {
        let bucket_time =
            now_unix_seconds() / TIMESERIES_BUCKET_SECONDS * TIMESERIES_BUCKET_SECONDS;
        if self
            .timeseries
            .back()
            .is_none_or(|bucket| bucket.unix_seconds != bucket_time)
        {
            push_bounded(
                &mut self.timeseries,
                TimeBucket {
                    unix_seconds: bucket_time,
                    ..TimeBucket::default()
                },
                MAX_TIMESERIES_BUCKETS,
            );
        }
        if let Some(bucket) = self.timeseries.back_mut() {
            bucket.requests = bucket.requests.saturating_add(1);
            if !record.status.is_success() {
                bucket.errors = bucket.errors.saturating_add(1);
            }
            match record.rejection {
                Some(GatewayRejectionKind::Blocked) => {
                    bucket.blocked = bucket.blocked.saturating_add(1)
                }
                Some(GatewayRejectionKind::ClientClosed) => {}
                Some(GatewayRejectionKind::RateLimited) => {
                    bucket.rate_limited = bucket.rate_limited.saturating_add(1)
                }
                None => {}
            }
            bucket.input_tokens = bucket.input_tokens.saturating_add(input_tokens);
            bucket.output_tokens = bucket.output_tokens.saturating_add(output_tokens);
            bucket.total_tokens = bucket.total_tokens.saturating_add(total_tokens);
            bucket.latency_ms_total = bucket.latency_ms_total.saturating_add(latency_ms);
            bucket.latency_samples = bucket.latency_samples.saturating_add(1);
        }
    }

    fn snapshot(
        &self,
        counters: [OperationMetricSnapshot; 3],
        active_requests: u64,
    ) -> GatewayDashboardSnapshot {
        let mut latency = self.latency_samples.iter().copied().collect::<Vec<_>>();
        latency.sort_unstable();
        GatewayDashboardSnapshot {
            totals: TotalsSnapshot {
                requests: self.total_requests,
                errors: self.total_errors,
                active_requests,
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                total_tokens: self.total_tokens,
                rate_limit_hits: self.rate_limit_hits,
                blocklist_hits: self.blocklist_hits,
                p50_latency_ms: percentile(&latency, 50),
                p90_latency_ms: percentile(&latency, 90),
                p99_latency_ms: percentile(&latency, 99),
            },
            operations: counters
                .into_iter()
                .map(|counter| {
                    let summary = self
                        .operations
                        .get(counter.operation)
                        .copied()
                        .unwrap_or_default();
                    OperationSnapshot {
                        operation: counter.operation,
                        requests: counter.requests,
                        errors: counter.errors,
                        input_tokens: summary.input_tokens,
                        output_tokens: summary.output_tokens,
                        total_tokens: summary.total_tokens,
                        average_latency_ms: average(summary.latency_ms_total, summary.requests),
                    }
                })
                .collect(),
            targets: self
                .targets
                .iter()
                .map(|(name, summary)| TargetSnapshot {
                    name: name.clone(),
                    requests: summary.requests,
                    errors: summary.errors,
                    input_tokens: summary.input_tokens,
                    output_tokens: summary.output_tokens,
                    total_tokens: summary.total_tokens,
                    average_latency_ms: average(summary.latency_ms_total, summary.requests),
                })
                .collect(),
            clients: top_clients(&self.clients),
            timeseries: self
                .timeseries
                .iter()
                .map(TimeBucketSnapshot::from)
                .collect(),
            recent: self.recent.iter().cloned().collect(),
        }
    }
}

/// Complete in-memory dashboard telemetry snapshot.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GatewayDashboardSnapshot {
    pub(crate) totals: TotalsSnapshot,
    pub(crate) operations: Vec<OperationSnapshot>,
    pub(crate) targets: Vec<TargetSnapshot>,
    pub(crate) clients: Vec<ClientSnapshot>,
    pub(crate) timeseries: Vec<TimeBucketSnapshot>,
    pub(crate) recent: Vec<RecentRequest>,
}

/// High-level request and token totals.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TotalsSnapshot {
    pub(crate) requests: u64,
    pub(crate) errors: u64,
    pub(crate) active_requests: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) rate_limit_hits: u64,
    pub(crate) blocklist_hits: u64,
    pub(crate) p50_latency_ms: u64,
    pub(crate) p90_latency_ms: u64,
    pub(crate) p99_latency_ms: u64,
}

/// Dashboard summary for one operation.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct OperationSnapshot {
    pub(crate) operation: &'static str,
    pub(crate) requests: u64,
    pub(crate) errors: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) average_latency_ms: u64,
}

/// Dashboard summary for one target.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TargetSnapshot {
    pub(crate) name: String,
    pub(crate) requests: u64,
    pub(crate) errors: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) average_latency_ms: u64,
}

/// Dashboard summary for one client IP.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ClientSnapshot {
    pub(crate) client: String,
    pub(crate) caller: Option<String>,
    pub(crate) requests: u64,
    pub(crate) errors: u64,
    pub(crate) rate_limit_hits: u64,
    pub(crate) blocklist_hits: u64,
    pub(crate) total_tokens: u64,
    pub(crate) last_seen_unix_seconds: u64,
    pub(crate) average_latency_ms: u64,
}

/// One dashboard time-series bucket.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TimeBucketSnapshot {
    pub(crate) unix_seconds: u64,
    pub(crate) requests: u64,
    pub(crate) errors: u64,
    pub(crate) rate_limited: u64,
    pub(crate) blocked: u64,
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) average_latency_ms: u64,
}

impl From<&TimeBucket> for TimeBucketSnapshot {
    fn from(bucket: &TimeBucket) -> Self {
        Self {
            unix_seconds: bucket.unix_seconds,
            requests: bucket.requests,
            errors: bucket.errors,
            rate_limited: bucket.rate_limited,
            blocked: bucket.blocked,
            input_tokens: bucket.input_tokens,
            output_tokens: bucket.output_tokens,
            total_tokens: bucket.total_tokens,
            average_latency_ms: average(bucket.latency_ms_total, bucket.latency_samples),
        }
    }
}

/// One recent request event.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RecentRequest {
    pub(crate) unix_seconds: u64,
    pub(crate) operation: &'static str,
    pub(crate) request_id: Option<String>,
    pub(crate) client: String,
    pub(crate) caller: Option<String>,
    pub(crate) target: Option<String>,
    pub(crate) status: u16,
    pub(crate) latency_ms: u64,
    pub(crate) total_tokens: u64,
    pub(crate) streaming: bool,
    pub(crate) rejection: Option<&'static str>,
}

impl RecentRequest {
    fn from_record(record: &GatewayRequestRecord, latency_ms: u64, total_tokens: u64) -> Self {
        Self {
            unix_seconds: now_unix_seconds(),
            operation: operation_name(record.operation),
            request_id: record.request_id.clone(),
            client: record.client.clone(),
            caller: record.caller.clone(),
            target: record.target.clone(),
            status: record.status.as_u16(),
            latency_ms,
            total_tokens,
            streaming: record.streaming,
            rejection: record.rejection.map(|kind| match kind {
                GatewayRejectionKind::Blocked => "blocked",
                GatewayRejectionKind::ClientClosed => "client_closed",
                GatewayRejectionKind::RateLimited => "rate_limited",
            }),
        }
    }
}

#[derive(Clone, Copy, Default)]
struct OperationSummary {
    requests: u64,
    errors: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    latency_ms_total: u64,
}

impl OperationSummary {
    fn record(
        &mut self,
        record: &GatewayRequestRecord,
        input_tokens: u64,
        output_tokens: u64,
        total_tokens: u64,
        latency_ms: u64,
    ) {
        self.requests = self.requests.saturating_add(1);
        if !record.status.is_success() {
            self.errors = self.errors.saturating_add(1);
        }
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
        self.total_tokens = self.total_tokens.saturating_add(total_tokens);
        self.latency_ms_total = self.latency_ms_total.saturating_add(latency_ms);
    }
}

type TargetSummary = OperationSummary;

#[derive(Clone, Default)]
struct ClientSummary {
    caller: Option<String>,
    requests: u64,
    errors: u64,
    rate_limit_hits: u64,
    blocklist_hits: u64,
    total_tokens: u64,
    latency_ms_total: u64,
    last_seen_unix_seconds: u64,
}

impl ClientSummary {
    fn record(
        &mut self,
        record: &GatewayRequestRecord,
        _input_tokens: u64,
        _output_tokens: u64,
        total_tokens: u64,
        latency_ms: u64,
    ) {
        self.requests = self.requests.saturating_add(1);
        if !record.status.is_success() {
            self.errors = self.errors.saturating_add(1);
        }
        match record.rejection {
            Some(GatewayRejectionKind::Blocked) => {
                self.blocklist_hits = self.blocklist_hits.saturating_add(1)
            }
            Some(GatewayRejectionKind::ClientClosed) => {}
            Some(GatewayRejectionKind::RateLimited) => {
                self.rate_limit_hits = self.rate_limit_hits.saturating_add(1)
            }
            None => {}
        }
        if record.caller.is_some() {
            self.caller = record.caller.clone();
        }
        self.total_tokens = self.total_tokens.saturating_add(total_tokens);
        self.latency_ms_total = self.latency_ms_total.saturating_add(latency_ms);
        self.last_seen_unix_seconds = now_unix_seconds();
    }
}

#[derive(Default)]
struct TimeBucket {
    unix_seconds: u64,
    requests: u64,
    errors: u64,
    rate_limited: u64,
    blocked: u64,
    input_tokens: u64,
    output_tokens: u64,
    total_tokens: u64,
    latency_ms_total: u64,
    latency_samples: u64,
}

fn push_bounded<T>(items: &mut VecDeque<T>, item: T, max_items: usize) {
    items.push_back(item);
    while items.len() > max_items {
        items.pop_front();
    }
}

fn prune_clients(clients: &mut BTreeMap<String, ClientSummary>) {
    while clients.len() > MAX_CLIENTS {
        let Some(oldest) = clients
            .iter()
            .min_by_key(|(_, summary)| summary.last_seen_unix_seconds)
            .map(|(client, _)| client.clone())
        else {
            break;
        };
        clients.remove(&oldest);
    }
}

fn top_clients(clients: &BTreeMap<String, ClientSummary>) -> Vec<ClientSnapshot> {
    let mut clients = clients
        .iter()
        .map(|(client, summary)| ClientSnapshot {
            client: client.clone(),
            caller: summary.caller.clone(),
            requests: summary.requests,
            errors: summary.errors,
            rate_limit_hits: summary.rate_limit_hits,
            blocklist_hits: summary.blocklist_hits,
            total_tokens: summary.total_tokens,
            last_seen_unix_seconds: summary.last_seen_unix_seconds,
            average_latency_ms: average(summary.latency_ms_total, summary.requests),
        })
        .collect::<Vec<_>>();
    clients.sort_by(|left, right| {
        right
            .requests
            .cmp(&left.requests)
            .then_with(|| right.total_tokens.cmp(&left.total_tokens))
    });
    clients.truncate(50);
    clients
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    if samples.is_empty() {
        return 0;
    }
    let index = ((samples.len() - 1) * percentile) / 100;
    samples[index]
}

fn average(total: u64, count: u64) -> u64 {
    if count == 0 {
        0
    } else {
        total / count
    }
}

fn millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

const fn operation_index(operation: Operation) -> usize {
    match operation {
        Operation::Query => 0,
        Operation::Chat => 1,
        Operation::Embed => 2,
    }
}

const fn operation_name(operation: Operation) -> &'static str {
    match operation {
        Operation::Query => "query",
        Operation::Chat => "chat",
        Operation::Embed => "embed",
    }
}

const fn status_class(status: StatusCode) -> &'static str {
    match status.as_u16() {
        100..=199 => "1xx",
        200..=299 => "2xx",
        300..=399 => "3xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        _ => "unknown",
    }
}
