//! Ephemeral runtime controls for the gateway-server application.

use std::collections::BTreeMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::http::HeaderMap;
use serde::Serialize;

use crate::config::{ClientIpConfig, ClientIpSource, RateLimitConfig, SecurityConfig};

const MAX_RATE_BUCKETS: usize = 4096;
const MAX_BLOCKED_CLIENTS: usize = 2048;
const RATE_BUCKET_IDLE_TTL: Duration = Duration::from_secs(15 * 60);

/// Runtime concurrency controls whose values disappear on restart.
#[derive(Clone)]
pub(crate) struct GatewayControls {
    concurrency: Arc<Mutex<ConcurrencyState>>,
}

impl GatewayControls {
    pub(crate) fn new(limit: Option<usize>) -> Self {
        Self {
            concurrency: Arc::new(Mutex::new(ConcurrencyState { limit, active: 0 })),
        }
    }

    pub(crate) fn try_acquire(&self) -> Result<ConcurrencyPermit, ConcurrencyAcquireError> {
        let Ok(mut state) = self.concurrency.lock() else {
            return Err(ConcurrencyAcquireError::Unavailable);
        };
        if state.limit.is_some_and(|limit| state.active >= limit) {
            return Err(ConcurrencyAcquireError::LimitExceeded);
        }
        state.active = state.active.saturating_add(1);
        Ok(ConcurrencyPermit {
            controls: self.clone(),
            released: false,
        })
    }

    pub(crate) fn set_concurrency_limit(&self, limit: Option<usize>) -> Result<(), &'static str> {
        if limit == Some(0) {
            return Err("concurrency limit must be greater than zero");
        }
        let mut state = self
            .concurrency
            .lock()
            .map_err(|_| "concurrency controls are unavailable")?;
        state.limit = limit;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> Result<ConcurrencySnapshot, &'static str> {
        let state = self
            .concurrency
            .lock()
            .map_err(|_| "concurrency controls are unavailable")?;
        Ok(ConcurrencySnapshot {
            limit: state.limit,
            active: state.active,
        })
    }

    fn release(&self) {
        let mut state = match self.concurrency.lock() {
            Ok(state) => state,
            Err(error) => error.into_inner(),
        };
        state.active = state.active.saturating_sub(1);
    }
}

/// Reason a request could not acquire a concurrency permit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConcurrencyAcquireError {
    LimitExceeded,
    Unavailable,
}

struct ConcurrencyState {
    limit: Option<usize>,
    active: usize,
}

/// Active runtime concurrency settings.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConcurrencySnapshot {
    pub(crate) limit: Option<usize>,
    pub(crate) active: usize,
}

/// Permit for one admitted request.
pub(crate) struct ConcurrencyPermit {
    controls: GatewayControls,
    released: bool,
}

impl Drop for ConcurrencyPermit {
    fn drop(&mut self) {
        if !self.released {
            self.controls.release();
            self.released = true;
        }
    }
}

/// Ephemeral in-memory security controls.
pub(crate) struct GatewaySecurity {
    client_ip: ClientIpRules,
    rate_limit: Mutex<RateLimitState>,
    blocklist: Mutex<BTreeMap<String, BlockEntry>>,
}

impl GatewaySecurity {
    pub(crate) fn new(config: &SecurityConfig) -> anyhow::Result<Self> {
        Ok(Self {
            client_ip: ClientIpRules::new(&config.client_ip)?,
            rate_limit: Mutex::new(RateLimitState::new(&config.rate_limit)),
            blocklist: Mutex::new(BTreeMap::new()),
        })
    }

    pub(crate) fn client_identity(&self, headers: &HeaderMap, peer: SocketAddr) -> String {
        self.client_ip.client_identity(headers, peer)
    }

    pub(crate) fn check_client(&self, client: &str) -> Result<(), SecurityCheckError> {
        if self.is_blocked(client)? {
            return Err(SecurityCheckError::Blocked);
        }
        self.check_rate_limit(client)
    }

    pub(crate) fn block_client(&self, client: &str) -> Result<SecuritySnapshot, &'static str> {
        let normalized = normalize_client(client)?;
        let mut blocklist = self
            .blocklist
            .lock()
            .map_err(|_| "blocklist is unavailable")?;
        blocklist.insert(
            normalized,
            BlockEntry {
                added_at: Instant::now(),
            },
        );
        prune_blocklist(&mut blocklist);
        drop(blocklist);
        self.snapshot()
    }

    pub(crate) fn unblock_client(&self, client: &str) -> Result<SecuritySnapshot, &'static str> {
        let normalized = normalize_client(client)?;
        let mut blocklist = self
            .blocklist
            .lock()
            .map_err(|_| "blocklist is unavailable")?;
        blocklist.remove(&normalized);
        drop(blocklist);
        self.snapshot()
    }

    pub(crate) fn update_rate_limit(
        &self,
        enabled: bool,
        requests_per_minute: u32,
        burst: u32,
    ) -> Result<SecuritySnapshot, &'static str> {
        if requests_per_minute == 0 {
            return Err("requests per minute must be greater than zero");
        }
        if burst == 0 {
            return Err("burst must be greater than zero");
        }
        let mut rate_limit = self
            .rate_limit
            .lock()
            .map_err(|_| "rate limiter is unavailable")?;
        *rate_limit = RateLimitState {
            enabled,
            requests_per_minute,
            burst,
            buckets: BTreeMap::new(),
        };
        drop(rate_limit);
        self.snapshot()
    }

    pub(crate) fn snapshot(&self) -> Result<SecuritySnapshot, &'static str> {
        let rate_limit = self
            .rate_limit
            .lock()
            .map(|state| state.snapshot())
            .map_err(|_| "rate limiter is unavailable")?;
        let blocklist = self
            .blocklist
            .lock()
            .map(|items| {
                let now = Instant::now();
                items
                    .iter()
                    .map(|(client, entry)| BlockedClientSnapshot {
                        client: client.clone(),
                        age_seconds: now.saturating_duration_since(entry.added_at).as_secs(),
                    })
                    .collect()
            })
            .map_err(|_| "blocklist is unavailable")?;
        Ok(SecuritySnapshot {
            client_ip_source: self.client_ip.source.as_str(),
            trusted_proxy_cidrs: self.client_ip.trusted_proxy_cidrs.clone(),
            rate_limit,
            blocklist,
        })
    }

    fn is_blocked(&self, client: &str) -> Result<bool, SecurityCheckError> {
        self.blocklist
            .lock()
            .map(|blocklist| blocklist.contains_key(client))
            .map_err(|_| SecurityCheckError::Unavailable)
    }

    fn check_rate_limit(&self, client: &str) -> Result<(), SecurityCheckError> {
        let Ok(mut state) = self.rate_limit.lock() else {
            return Err(SecurityCheckError::Unavailable);
        };
        if !state.enabled {
            return Ok(());
        }
        let now = Instant::now();
        state.prune(now);
        let refill_per_second = f64::from(state.requests_per_minute) / 60.0;
        let burst = f64::from(state.burst);
        let bucket = state
            .buckets
            .entry(client.to_string())
            .or_insert(TokenBucket {
                tokens: burst,
                updated_at: now,
                last_seen: now,
            });
        let elapsed = now
            .saturating_duration_since(bucket.updated_at)
            .as_secs_f64();
        bucket.tokens = (bucket.tokens + elapsed * refill_per_second).min(burst);
        bucket.updated_at = now;
        bucket.last_seen = now;
        if bucket.tokens < 1.0 {
            return Err(SecurityCheckError::RateLimited);
        }
        bucket.tokens -= 1.0;
        Ok(())
    }
}

/// Security rejection category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SecurityCheckError {
    Blocked,
    RateLimited,
    Unavailable,
}

/// Current in-memory security settings.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SecuritySnapshot {
    pub(crate) client_ip_source: &'static str,
    pub(crate) trusted_proxy_cidrs: Vec<String>,
    pub(crate) rate_limit: RateLimitSnapshot,
    pub(crate) blocklist: Vec<BlockedClientSnapshot>,
}

/// Current in-memory rate limiter settings.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RateLimitSnapshot {
    pub(crate) enabled: bool,
    pub(crate) requests_per_minute: u32,
    pub(crate) burst: u32,
    pub(crate) tracked_clients: usize,
}

/// One manually blocked client.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct BlockedClientSnapshot {
    pub(crate) client: String,
    pub(crate) age_seconds: u64,
}

struct RateLimitState {
    enabled: bool,
    requests_per_minute: u32,
    burst: u32,
    buckets: BTreeMap<String, TokenBucket>,
}

impl RateLimitState {
    fn new(config: &RateLimitConfig) -> Self {
        Self {
            enabled: config.enabled,
            requests_per_minute: config.requests_per_minute,
            burst: config.burst,
            buckets: BTreeMap::new(),
        }
    }

    fn snapshot(&self) -> RateLimitSnapshot {
        RateLimitSnapshot {
            enabled: self.enabled,
            requests_per_minute: self.requests_per_minute,
            burst: self.burst,
            tracked_clients: self.buckets.len(),
        }
    }

    fn prune(&mut self, now: Instant) {
        self.buckets.retain(|_, bucket| {
            now.saturating_duration_since(bucket.last_seen) <= RATE_BUCKET_IDLE_TTL
        });
        while self.buckets.len() > MAX_RATE_BUCKETS {
            let Some(first) = self.buckets.keys().next().cloned() else {
                break;
            };
            self.buckets.remove(&first);
        }
    }
}

struct TokenBucket {
    tokens: f64,
    updated_at: Instant,
    last_seen: Instant,
}

struct BlockEntry {
    added_at: Instant,
}

fn prune_blocklist(blocklist: &mut BTreeMap<String, BlockEntry>) {
    while blocklist.len() > MAX_BLOCKED_CLIENTS {
        let Some(first) = blocklist.keys().next().cloned() else {
            break;
        };
        blocklist.remove(&first);
    }
}

fn normalize_client(client: &str) -> Result<String, &'static str> {
    client
        .parse::<IpAddr>()
        .map(|ip| ip.to_string())
        .map_err(|_| "client must be an IP address")
}

struct ClientIpRules {
    source: ClientIpSource,
    trusted_proxy_cidrs: Vec<String>,
    trusted_proxies: Vec<IpTrustRule>,
}

impl ClientIpRules {
    fn new(config: &ClientIpConfig) -> anyhow::Result<Self> {
        Ok(Self {
            source: config.source,
            trusted_proxy_cidrs: config.trusted_proxy_cidrs.clone(),
            trusted_proxies: config
                .trusted_proxy_cidrs
                .iter()
                .map(|cidr| IpTrustRule::parse(cidr))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    fn client_identity(&self, headers: &HeaderMap, peer: SocketAddr) -> String {
        let peer_ip = peer.ip();
        if self.source != ClientIpSource::Peer && self.forwarded_headers_allowed(peer_ip) {
            if let Some(ip) = self.header_client_ip(headers) {
                return ip.to_string();
            }
        }
        peer_ip.to_string()
    }

    fn forwarded_headers_allowed(&self, peer: IpAddr) -> bool {
        if self.trusted_proxies.is_empty() {
            return false;
        }
        self.trusted_proxies.iter().any(|rule| rule.contains(peer))
    }

    fn header_client_ip(&self, headers: &HeaderMap) -> Option<IpAddr> {
        match self.source {
            ClientIpSource::Peer => None,
            ClientIpSource::XForwardedFor => headers
                .get("x-forwarded-for")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.split(',').next())
                .map(str::trim)
                .and_then(|value| value.parse().ok()),
            ClientIpSource::XRealIp => headers
                .get("x-real-ip")
                .and_then(|value| value.to_str().ok())
                .and_then(|value| value.trim().parse().ok()),
        }
    }
}

#[derive(Clone)]
struct IpTrustRule {
    base: IpAddr,
    prefix: u8,
}

impl IpTrustRule {
    fn parse(value: &str) -> anyhow::Result<Self> {
        let (address, prefix) = value
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("trusted proxy CIDR must include a prefix: {value}"))?;
        Ok(Self {
            base: address.parse()?,
            prefix: prefix.parse()?,
        })
    }

    fn contains(&self, ip: IpAddr) -> bool {
        match (self.base, ip) {
            (IpAddr::V4(base), IpAddr::V4(ip)) => contains_v4(base, ip, self.prefix),
            (IpAddr::V6(base), IpAddr::V6(ip)) => contains_v6(base, ip, self.prefix),
            _ => false,
        }
    }
}

fn contains_v4(base: Ipv4Addr, ip: Ipv4Addr, prefix: u8) -> bool {
    let base = u32::from(base);
    let ip = u32::from(ip);
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    };
    base & mask == ip & mask
}

fn contains_v6(base: Ipv6Addr, ip: Ipv6Addr, prefix: u8) -> bool {
    let base = u128::from(base);
    let ip = u128::from(ip);
    let mask = if prefix == 0 {
        0
    } else {
        u128::MAX << (128 - u32::from(prefix))
    };
    base & mask == ip & mask
}
