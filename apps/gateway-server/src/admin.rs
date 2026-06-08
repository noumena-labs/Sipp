use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::header::{COOKIE, LOCATION, SET_COOKIE};
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use cogentlm_engine::backend::backend_observability_json;
use rand::random;

use crate::config::{admin_login_route, admin_logout_route, RouteConfig, TargetSummary};
use crate::metrics::GatewayMetrics;

const SESSION_COOKIE: &str = "cogentlm_gateway_admin";
const SESSION_TTL: Duration = Duration::from_secs(8 * 60 * 60);

#[derive(Clone)]
pub(crate) struct AdminDashboardState {
    password: Arc<String>,
    sessions: Arc<Mutex<BTreeMap<String, Instant>>>,
    view: Arc<AdminDashboardView>,
    metrics: Arc<GatewayMetrics>,
}

impl AdminDashboardState {
    pub(crate) fn new(
        password: String,
        view: AdminDashboardView,
        metrics: Arc<GatewayMetrics>,
    ) -> Self {
        Self {
            password: Arc::new(password),
            sessions: Arc::new(Mutex::new(BTreeMap::new())),
            view: Arc::new(view),
            metrics,
        }
    }
}

#[derive(Clone)]
pub(crate) struct AdminDashboardView {
    pub(crate) routes: RouteConfig,
    pub(crate) targets: Vec<TargetSummary>,
    pub(crate) max_request_bytes: usize,
    pub(crate) max_concurrent_requests: Option<usize>,
    pub(crate) started_at: Instant,
}

pub(crate) fn router(route: &str, state: AdminDashboardState) -> Router {
    Router::new()
        .route(route, get(dashboard))
        .route(&admin_login_route(route), post(login))
        .route(&admin_logout_route(route), post(logout))
        .with_state(state)
}

async fn dashboard(State(state): State<AdminDashboardState>, headers: HeaderMap) -> Response<Body> {
    if !state.authenticated(&headers) {
        return html_response(StatusCode::OK, login_html(&state.view, false), None);
    }

    html_response(
        StatusCode::OK,
        dashboard_html(&state.view, &state.metrics),
        None,
    )
}

async fn login(State(state): State<AdminDashboardState>, body: Bytes) -> Response<Body> {
    let password = form_value(&body, "password").unwrap_or_default();
    if !constant_time_eq(&password, &state.password) {
        return html_response(
            StatusCode::UNAUTHORIZED,
            login_html(&state.view, true),
            None,
        );
    }

    let session = state.create_session();
    redirect_response(
        state.view.routes.admin.as_deref().unwrap_or("/admin"),
        Some(session_cookie(&session)),
    )
}

async fn logout(State(state): State<AdminDashboardState>, headers: HeaderMap) -> Response<Body> {
    if let Some(session) = session_cookie_value(&headers) {
        state.remove_session(session);
    }
    redirect_response(
        state.view.routes.admin.as_deref().unwrap_or("/admin"),
        Some(clear_session_cookie()),
    )
}

impl AdminDashboardState {
    fn authenticated(&self, headers: &HeaderMap) -> bool {
        let Some(session) = session_cookie_value(headers) else {
            return false;
        };
        let Ok(mut sessions) = self.sessions.lock() else {
            return false;
        };
        prune_sessions(&mut sessions);
        sessions
            .get(session)
            .is_some_and(|expires_at| *expires_at > Instant::now())
    }

    fn create_session(&self) -> String {
        let session = random_session_id();
        if let Ok(mut sessions) = self.sessions.lock() {
            prune_sessions(&mut sessions);
            sessions.insert(session.clone(), Instant::now() + SESSION_TTL);
        }
        session
    }

    fn remove_session(&self, session: &str) {
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.remove(session);
        }
    }
}

fn prune_sessions(sessions: &mut BTreeMap<String, Instant>) {
    let now = Instant::now();
    sessions.retain(|_, expires_at| *expires_at > now);
}

fn random_session_id() -> String {
    hex(&random::<[u8; 32]>())
}

fn html_response(status: StatusCode, body: String, cookie: Option<String>) -> Response<Body> {
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", "text/html; charset=utf-8")
        .header("cache-control", "no-store");
    if let Some(cookie) = cookie.and_then(|value| HeaderValue::from_str(&value).ok()) {
        builder = builder.header(SET_COOKIE, cookie);
    }
    builder.body(Body::from(body)).unwrap_or_else(|_| {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        response
    })
}

fn redirect_response(location: &str, cookie: Option<String>) -> Response<Body> {
    let mut builder = Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header(LOCATION, location)
        .header("cache-control", "no-store");
    if let Some(cookie) = cookie.and_then(|value| HeaderValue::from_str(&value).ok()) {
        builder = builder.header(SET_COOKIE, cookie);
    }
    builder.body(Body::empty()).unwrap_or_else(|_| {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
        response
    })
}

fn login_html(view: &AdminDashboardView, failed: bool) -> String {
    let error = if failed {
        r#"<p class="error">Invalid password.</p>"#
    } else {
        ""
    };
    let login_route = view
        .routes
        .admin
        .as_deref()
        .map(admin_login_route)
        .unwrap_or_else(|| "/admin/login".to_string());
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CogentLM Gateway Admin</title>
<style>{style}</style>
</head>
<body>
<main class="login">
<section class="panel">
<h1>Gateway Admin</h1>
{error}
<form method="post" action="{login_route}">
<label>Password<input type="password" name="password" autofocus autocomplete="current-password"></label>
<button type="submit">Sign in</button>
</form>
</section>
</main>
</body>
</html>"#,
        style = STYLE,
        error = error,
        login_route = html_escape(&login_route)
    )
}

fn dashboard_html(view: &AdminDashboardView, metrics: &GatewayMetrics) -> String {
    let uptime = human_duration(view.started_at.elapsed());
    let backend_json = backend_observability_json(true).unwrap_or_else(|_| "{}".to_string());
    let backend_value = serde_json::from_str::<serde_json::Value>(&backend_json)
        .unwrap_or_else(|_| serde_json::json!({}));
    let metrics = metrics.snapshot();
    let target_rows = target_rows(&view.targets);
    let metric_rows = metrics
        .iter()
        .map(|metric| {
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td></tr>",
                html_escape(metric.operation),
                metric.requests,
                metric.errors
            )
        })
        .collect::<String>();
    let backend_rows = backend_rows(&backend_value);
    let route_rows = route_rows(&view.routes);

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CogentLM Gateway Admin</title>
<style>{style}</style>
</head>
<body>
<header>
<div>
<h1>Gateway Admin</h1>
<p>Uptime {uptime}</p>
</div>
<form method="post" action="{logout_route}"><button type="submit">Sign out</button></form>
</header>
<main>
<section>
<h2>Status</h2>
<div class="stats">
<div><strong>Health</strong><span>ok</span></div>
<div><strong>Readiness</strong><span>ready</span></div>
<div><strong>Max request bytes</strong><span>{max_request_bytes}</span></div>
<div><strong>Concurrency limit</strong><span>{max_concurrent_requests}</span></div>
</div>
</section>
<section>
<h2>Targets</h2>
<table><thead><tr><th>Name</th><th>Kind</th><th>Model</th><th>Backend</th><th>Provider URL</th></tr></thead><tbody>{target_rows}</tbody></table>
</section>
<section>
<h2>Metrics</h2>
<table><thead><tr><th>Operation</th><th>Requests</th><th>Errors</th></tr></thead><tbody>{metric_rows}</tbody></table>
</section>
<section>
<h2>Backends</h2>
<table><thead><tr><th>Name</th><th>Devices</th></tr></thead><tbody>{backend_rows}</tbody></table>
</section>
<section>
<h2>Routes</h2>
<table><thead><tr><th>Name</th><th>Path</th></tr></thead><tbody>{route_rows}</tbody></table>
</section>
<section>
<h2>Runtime</h2>
<p class="muted">Admin password: configured in TOML</p>
</section>
</main>
</body>
</html>"#,
        style = STYLE,
        uptime = html_escape(&uptime),
        logout_route = html_escape(
            &view
                .routes
                .admin
                .as_deref()
                .map(admin_logout_route)
                .unwrap_or_else(|| "/admin/logout".to_string())
        ),
        max_request_bytes = view.max_request_bytes,
        max_concurrent_requests = view
            .max_concurrent_requests
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unbounded".to_string()),
        target_rows = target_rows,
        metric_rows = metric_rows,
        backend_rows = backend_rows,
        route_rows = route_rows,
    )
}

fn target_rows(targets: &[TargetSummary]) -> String {
    targets
        .iter()
        .map(|target| {
            let backend = target
                .backend
                .as_ref()
                .map(|backend| {
                    format!(
                        "{} ({})",
                        backend.selected,
                        backend.reason.as_deref().unwrap_or("selected")
                    )
                })
                .unwrap_or_else(|| "provider".to_string());
            format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                html_escape(&target.name),
                html_escape(target.kind.as_str()),
                html_escape(&target.model),
                html_escape(&backend),
                html_escape(target.provider_base_url.as_deref().unwrap_or("-"))
            )
        })
        .collect()
}

fn backend_rows(value: &serde_json::Value) -> String {
    value
        .get("availableBackends")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|item| {
                    let name = item
                        .get("name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown");
                    let device_count = item
                        .get("deviceCount")
                        .and_then(serde_json::Value::as_u64)
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    format!(
                        "<tr><td>{}</td><td>{}</td></tr>",
                        html_escape(name),
                        html_escape(&device_count)
                    )
                })
                .collect()
        })
        .unwrap_or_else(|| "<tr><td>unknown</td><td>-</td></tr>".to_string())
}

fn route_rows(routes: &RouteConfig) -> String {
    [
        ("query", Some(routes.query.as_str())),
        ("chat", Some(routes.chat.as_str())),
        ("embed", Some(routes.embed.as_str())),
        ("index", routes.index.as_deref()),
        ("health", routes.health.as_deref()),
        ("readiness", routes.readiness.as_deref()),
        ("metrics", routes.metrics.as_deref()),
        ("admin", routes.admin.as_deref()),
    ]
    .into_iter()
    .filter_map(|(name, route)| route.map(|route| (name, route)))
    .map(|(name, route)| {
        format!(
            "<tr><td>{}</td><td>{}</td></tr>",
            html_escape(name),
            html_escape(route)
        )
    })
    .collect()
}

fn session_cookie(session: &str) -> String {
    format!(
        "{SESSION_COOKIE}={session}; Max-Age={}; Path=/; HttpOnly; SameSite=Lax",
        SESSION_TTL.as_secs()
    )
}

fn clear_session_cookie() -> String {
    format!("{SESSION_COOKIE}=; Max-Age=0; Path=/; HttpOnly; SameSite=Lax")
}

fn session_cookie_value(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(COOKIE)
        .and_then(|value| value.to_str().ok())
        .and_then(|header| {
            header.split(';').find_map(|part| {
                let (name, value) = part.trim().split_once('=')?;
                (name == SESSION_COOKIE).then_some(value)
            })
        })
}

fn form_value(body: &[u8], key: &str) -> Option<String> {
    let body = std::str::from_utf8(body).ok()?;
    body.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (percent_decode(name) == key).then(|| percent_decode(value))
    })
}

fn percent_decode(value: &str) -> String {
    let mut bytes = Vec::with_capacity(value.len());
    let mut input = value.as_bytes().iter().copied();
    while let Some(byte) = input.next() {
        match byte {
            b'+' => bytes.push(b' '),
            b'%' => {
                let hi = input.next();
                let lo = input.next();
                match (hi.and_then(hex_value), lo.and_then(hex_value)) {
                    (Some(hi), Some(lo)) => bytes.push((hi << 4) | lo),
                    _ => {
                        bytes.push(b'%');
                        if let Some(hi) = hi {
                            bytes.push(hi);
                        }
                        if let Some(lo) = lo {
                            bytes.push(lo);
                        }
                    }
                }
            }
            _ => bytes.push(byte),
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn html_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            '\'' => output.push_str("&#39;"),
            _ => output.push(ch),
        }
    }
    output
}

fn human_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn constant_time_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left = left.get(index).copied().unwrap_or(0);
        let right = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(left ^ right);
    }
    difference == 0
}

const STYLE: &str = r#"
:root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1d252c; background: #f6f7f8; }
body { margin: 0; }
header { align-items: center; background: #ffffff; border-bottom: 1px solid #d9dee3; display: flex; justify-content: space-between; padding: 18px 32px; }
header h1, .panel h1 { font-size: 22px; margin: 0 0 4px; }
header p { color: #5d6873; margin: 0; }
main { display: grid; gap: 20px; margin: 0 auto; max-width: 1180px; padding: 24px; }
section, .panel { background: #ffffff; border: 1px solid #d9dee3; border-radius: 8px; padding: 18px; }
h2 { font-size: 16px; margin: 0 0 14px; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #e7eaee; padding: 10px 8px; text-align: left; vertical-align: top; }
th { color: #53606b; font-size: 12px; text-transform: uppercase; }
button { background: #1f6feb; border: 0; border-radius: 6px; color: #ffffff; cursor: pointer; font-weight: 600; padding: 9px 14px; }
.login { min-height: 100vh; place-items: center; }
.login .panel { width: min(420px, calc(100vw - 48px)); }
label { display: grid; font-weight: 600; gap: 8px; }
input { border: 1px solid #bac3cc; border-radius: 6px; font: inherit; padding: 10px; }
.panel form { display: grid; gap: 14px; }
.error { color: #b42318; font-weight: 600; }
.stats { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.stats div { border: 1px solid #e1e5e9; border-radius: 6px; padding: 12px; }
.stats strong { color: #53606b; display: block; font-size: 12px; margin-bottom: 6px; text-transform: uppercase; }
.muted { color: #5d6873; margin: 0; }
"#;
