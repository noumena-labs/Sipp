use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::header::{COOKIE, LOCATION, SET_COOKIE};
use axum::http::{HeaderMap, HeaderValue, Response, StatusCode};
use axum::routing::{get, put};
use axum::{Json, Router};
use sipp::backend::backend_observability_json;
use rand::random;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::{RouteConfig, TargetSummary};
use crate::metrics::GatewayMetrics;
use crate::runtime::{GatewayControls, GatewaySecurity};

const CSRF_HEADER: &str = "x-sipp-admin-csrf";
const SESSION_COOKIE: &str = "sipp_gateway_admin";
const SESSION_TTL: Duration = Duration::from_secs(8 * 60 * 60);

#[derive(Clone)]
pub(crate) struct AdminDashboardState {
    password: Arc<String>,
    sessions: Arc<Mutex<BTreeMap<String, AdminSession>>>,
    view: Arc<AdminDashboardView>,
    metrics: Arc<GatewayMetrics>,
    controls: Arc<GatewayControls>,
    security: Arc<GatewaySecurity>,
    assets: Arc<AdminAssets>,
}

impl AdminDashboardState {
    pub(crate) fn new(
        password: String,
        view: AdminDashboardView,
        metrics: Arc<GatewayMetrics>,
        controls: Arc<GatewayControls>,
        security: Arc<GatewaySecurity>,
        assets_dir: PathBuf,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            password: Arc::new(password),
            sessions: Arc::new(Mutex::new(BTreeMap::new())),
            view: Arc::new(view),
            metrics,
            controls,
            security,
            assets: Arc::new(AdminAssets::new(assets_dir)?),
        })
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

impl AdminDashboardView {
    fn admin_base(&self) -> String {
        self.routes
            .admin
            .as_deref()
            .map(admin_base_path)
            .unwrap_or_else(|| "/admin".to_string())
    }

    fn admin_base_with_slash(&self) -> String {
        admin_base_with_slash(&self.admin_base())
    }
}

#[derive(Clone)]
struct AdminSession {
    expires_at: Instant,
    csrf_token: String,
}

struct AuthenticatedSession {
    csrf_token: String,
}

struct AdminAssets {
    root: PathBuf,
}

impl AdminAssets {
    fn new(root: PathBuf) -> anyhow::Result<Self> {
        let index = root.join("index.html");
        if !index.is_file() {
            anyhow::bail!(
                "gateway Admin Dashboard assets are missing; expected {}",
                index.display()
            );
        }
        Ok(Self { root })
    }

    async fn read(&self, path: &str) -> Result<Option<AdminAsset>, AdminApiError> {
        let Some(relative) = clean_asset_path(path) else {
            return Ok(None);
        };
        let path = self.root.join(relative);
        match tokio::fs::read(&path).await {
            Ok(bytes) => Ok(Some(AdminAsset {
                content_type: content_type(&path),
                bytes,
            })),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(AdminApiError::internal("failed to read dashboard asset")),
        }
    }
}

struct AdminAsset {
    content_type: &'static str,
    bytes: Vec<u8>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionResponse {
    authenticated: bool,
    csrf_token: String,
    csrf_header: &'static str,
    base_path: String,
}

#[derive(Clone)]
struct AdminApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

impl AdminApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            code: "unauthorized",
            message: "admin session is required".to_string(),
        }
    }

    fn forbidden(code: &'static str, message: &'static str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            code,
            message: message.to_string(),
        }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            code: "bad_request",
            message: message.into(),
        }
    }

    fn internal(message: &'static str) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "internal",
            message: message.to_string(),
        }
    }

    fn into_response(self) -> Response<Body> {
        json_response(
            self.status,
            &json!({
                "error": self.code,
                "message": self.message,
            }),
            None,
        )
    }
}

pub(crate) fn router(route: &str, state: AdminDashboardState) -> Router {
    let root = admin_base_path(route);
    let root_with_slash = admin_base_with_slash(&root);
    let api_session_route = join_admin_route(&root, "api/session");
    let api_overview_route = join_admin_route(&root, "api/overview");
    let api_timeseries_route = join_admin_route(&root, "api/timeseries");
    let api_routes_route = join_admin_route(&root, "api/routes");
    let api_targets_route = join_admin_route(&root, "api/targets");
    let api_clients_route = join_admin_route(&root, "api/clients");
    let api_security_route = join_admin_route(&root, "api/security");
    let api_rate_limit_route = join_admin_route(&root, "api/security/rate-limit");
    let api_blocklist_route = join_admin_route(&root, "api/security/blocklist/{client}");
    let api_concurrency_route = join_admin_route(&root, "api/controls/concurrency");
    let wildcard = join_admin_route(&root, "{*path}");

    let mut router = Router::new()
        .route(
            &api_session_route,
            get(api_session)
                .post(api_create_session)
                .delete(api_delete_session),
        )
        .route(&api_overview_route, get(api_overview))
        .route(&api_timeseries_route, get(api_timeseries))
        .route(&api_routes_route, get(api_routes))
        .route(&api_targets_route, get(api_targets))
        .route(&api_clients_route, get(api_clients))
        .route(&api_security_route, get(api_security))
        .route(&api_rate_limit_route, put(api_update_rate_limit))
        .route(
            &api_blocklist_route,
            get(api_not_found)
                .post(api_block_client)
                .delete(api_unblock_client),
        )
        .route(&api_concurrency_route, put(api_update_concurrency))
        .route(&root_with_slash, get(dashboard_index))
        .route(&wildcard, get(dashboard_path));

    if root != root_with_slash {
        router = router.route(&root, get(redirect_to_dashboard_root));
    }

    router.with_state(state)
}

async fn redirect_to_dashboard_root(State(state): State<AdminDashboardState>) -> Response<Body> {
    redirect_response(&state.view.admin_base_with_slash(), None)
}

async fn dashboard_index(State(state): State<AdminDashboardState>) -> Response<Body> {
    match asset_response(&state, "index.html").await {
        Ok(Some(response)) => response,
        Ok(None) => AdminApiError::internal("dashboard index asset is missing").into_response(),
        Err(error) => error.into_response(),
    }
}

async fn dashboard_path(
    State(state): State<AdminDashboardState>,
    AxumPath(path): AxumPath<String>,
) -> Response<Body> {
    if path.starts_with("api/") {
        return api_not_found().await;
    }

    match asset_response(&state, &path).await {
        Ok(Some(response)) => response,
        Ok(None) if path.starts_with("assets/") => empty_response(StatusCode::NOT_FOUND, None),
        Ok(None) => dashboard_index(State(state)).await,
        Err(error) => error.into_response(),
    }
}

async fn api_session(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    match state.api_session(&headers) {
        Ok(session) => session_response(&state, session, None),
        Err(error) => error.into_response(),
    }
}

async fn api_create_session(
    State(state): State<AdminDashboardState>,
    Json(payload): Json<LoginPayload>,
) -> Response<Body> {
    if !constant_time_eq(&payload.password, &state.password) {
        return AdminApiError::unauthorized().into_response();
    }

    match state.create_session() {
        Ok((session, authenticated)) => {
            session_response(&state, authenticated, Some(session_cookie(&session)))
        }
        Err(error) => error.into_response(),
    }
}

async fn api_delete_session(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Some(session) = session_cookie_value(&headers) {
        if let Err(error) = state.remove_session(session) {
            return error.into_response();
        }
    }
    json_response(
        StatusCode::OK,
        &json!({ "authenticated": false }),
        Some(clear_session_cookie()),
    )
}

async fn api_overview(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    let backend_json = match backend_observability_json(true) {
        Ok(value) => value,
        Err(_) => {
            return AdminApiError::internal("failed to collect backend observability")
                .into_response()
        }
    };
    let backend_value = match serde_json::from_str::<serde_json::Value>(&backend_json) {
        Ok(value) => value,
        Err(_) => {
            return AdminApiError::internal("failed to parse backend observability").into_response()
        }
    };
    let metrics = match state.metrics.dashboard_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    let concurrency = match state.controls.snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    let security = match state.security.snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    json_response(
        StatusCode::OK,
        &json!({
            "uptimeSeconds": state.view.started_at.elapsed().as_secs(),
            "maxRequestBytes": state.view.max_request_bytes,
            "configuredConcurrencyLimit": state.view.max_concurrent_requests,
            "metrics": metrics,
            "controls": {
                "concurrency": concurrency,
            },
            "security": security,
            "backends": backend_value,
        }),
        None,
    )
}

async fn api_timeseries(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    let snapshot = match state.metrics.dashboard_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    json_response(
        StatusCode::OK,
        &json!({ "timeseries": snapshot.timeseries }),
        None,
    )
}

async fn api_routes(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    json_response(
        StatusCode::OK,
        &json!({ "routes": route_values(&state.view.routes) }),
        None,
    )
}

async fn api_targets(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    let snapshot = match state.metrics.dashboard_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    json_response(
        StatusCode::OK,
        &json!({
            "targets": state.view.targets.iter().map(target_value).collect::<Vec<_>>(),
            "metrics": snapshot.targets,
        }),
        None,
    )
}

async fn api_clients(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    let snapshot = match state.metrics.dashboard_snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => return AdminApiError::internal(error).into_response(),
    };
    json_response(
        StatusCode::OK,
        &json!({
            "clients": snapshot.clients,
            "recent": snapshot.recent,
        }),
        None,
    )
}

async fn api_security(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
) -> Response<Body> {
    if let Err(error) = state.api_session(&headers) {
        return error.into_response();
    }
    match state.security.snapshot() {
        Ok(snapshot) => json_response(StatusCode::OK, &json!({ "security": snapshot }), None),
        Err(error) => AdminApiError::internal(error).into_response(),
    }
}

async fn api_update_rate_limit(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
    Json(payload): Json<RateLimitPayload>,
) -> Response<Body> {
    if let Err(error) = state.require_csrf(&headers) {
        return error.into_response();
    }
    match state.security.update_rate_limit(
        payload.enabled,
        payload.requests_per_minute,
        payload.burst,
    ) {
        Ok(snapshot) => json_response(StatusCode::OK, &json!({ "security": snapshot }), None),
        Err(error) => control_error_response(error),
    }
}

async fn api_block_client(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
    AxumPath(client): AxumPath<String>,
) -> Response<Body> {
    if let Err(error) = state.require_csrf(&headers) {
        return error.into_response();
    }
    match state.security.block_client(&client) {
        Ok(snapshot) => json_response(StatusCode::OK, &json!({ "security": snapshot }), None),
        Err(error) => control_error_response(error),
    }
}

async fn api_unblock_client(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
    AxumPath(client): AxumPath<String>,
) -> Response<Body> {
    if let Err(error) = state.require_csrf(&headers) {
        return error.into_response();
    }
    match state.security.unblock_client(&client) {
        Ok(snapshot) => json_response(StatusCode::OK, &json!({ "security": snapshot }), None),
        Err(error) => control_error_response(error),
    }
}

async fn api_update_concurrency(
    State(state): State<AdminDashboardState>,
    headers: HeaderMap,
    Json(payload): Json<ConcurrencyPayload>,
) -> Response<Body> {
    if let Err(error) = state.require_csrf(&headers) {
        return error.into_response();
    }
    match state.controls.set_concurrency_limit(payload.limit) {
        Ok(()) => match state.controls.snapshot() {
            Ok(snapshot) => {
                json_response(StatusCode::OK, &json!({ "concurrency": snapshot }), None)
            }
            Err(error) => AdminApiError::internal(error).into_response(),
        },
        Err(error) => control_error_response(error),
    }
}

async fn api_not_found() -> Response<Body> {
    json_response(
        StatusCode::NOT_FOUND,
        &json!({ "error": "not_found" }),
        None,
    )
}

#[derive(Deserialize)]
struct LoginPayload {
    password: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RateLimitPayload {
    enabled: bool,
    requests_per_minute: u32,
    burst: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ConcurrencyPayload {
    limit: Option<usize>,
}

impl AdminDashboardState {
    fn api_session(&self, headers: &HeaderMap) -> Result<AuthenticatedSession, AdminApiError> {
        let Some(session) = session_cookie_value(headers) else {
            return Err(AdminApiError::unauthorized());
        };
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| AdminApiError::internal("admin session state is unavailable"))?;
        prune_sessions(&mut sessions);
        sessions
            .get(session)
            .filter(|session| session.expires_at > Instant::now())
            .map(|session| AuthenticatedSession {
                csrf_token: session.csrf_token.clone(),
            })
            .ok_or_else(AdminApiError::unauthorized)
    }

    fn require_csrf(&self, headers: &HeaderMap) -> Result<AuthenticatedSession, AdminApiError> {
        let session = self.api_session(headers)?;
        let supplied = headers
            .get(CSRF_HEADER)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        if !constant_time_eq(supplied, &session.csrf_token) {
            return Err(AdminApiError::forbidden(
                "csrf_failed",
                "admin CSRF token is invalid",
            ));
        }
        Ok(session)
    }

    fn create_session(&self) -> Result<(String, AuthenticatedSession), AdminApiError> {
        let session = random_session_id();
        let csrf_token = random_session_id();
        let mut sessions = self
            .sessions
            .lock()
            .map_err(|_| AdminApiError::internal("admin session state is unavailable"))?;
        prune_sessions(&mut sessions);
        sessions.insert(
            session.clone(),
            AdminSession {
                expires_at: Instant::now() + SESSION_TTL,
                csrf_token: csrf_token.clone(),
            },
        );
        Ok((session, AuthenticatedSession { csrf_token }))
    }

    fn remove_session(&self, session: &str) -> Result<(), AdminApiError> {
        self.sessions
            .lock()
            .map_err(|_| AdminApiError::internal("admin session state is unavailable"))?
            .remove(session);
        Ok(())
    }
}

fn session_response(
    state: &AdminDashboardState,
    session: AuthenticatedSession,
    cookie: Option<String>,
) -> Response<Body> {
    json_response(
        StatusCode::OK,
        &SessionResponse {
            authenticated: true,
            csrf_token: session.csrf_token,
            csrf_header: CSRF_HEADER,
            base_path: state.view.admin_base(),
        },
        cookie,
    )
}

fn control_error_response(error: &'static str) -> Response<Body> {
    if error.contains("unavailable") {
        AdminApiError::internal(error).into_response()
    } else {
        AdminApiError::bad_request(error).into_response()
    }
}

fn prune_sessions(sessions: &mut BTreeMap<String, AdminSession>) {
    let now = Instant::now();
    sessions.retain(|_, session| session.expires_at > now);
}

fn random_session_id() -> String {
    hex(&random::<[u8; 32]>())
}

async fn asset_response(
    state: &AdminDashboardState,
    path: &str,
) -> Result<Option<Response<Body>>, AdminApiError> {
    state.assets.read(path).await.map(|asset| {
        asset.map(|asset| {
            response_with_body(
                StatusCode::OK,
                asset.content_type,
                Body::from(asset.bytes),
                None,
            )
        })
    })
}

fn json_response<T: Serialize>(
    status: StatusCode,
    value: &T,
    cookie: Option<String>,
) -> Response<Body> {
    match serde_json::to_vec(value) {
        Ok(body) => response_with_body(status, "application/json", Body::from(body), cookie),
        Err(_) => response_with_body(
            StatusCode::INTERNAL_SERVER_ERROR,
            "application/json",
            Body::from(r#"{"error":"encoding_failed"}"#),
            cookie,
        ),
    }
}

fn response_with_body(
    status: StatusCode,
    content_type: &'static str,
    body: Body,
    cookie: Option<String>,
) -> Response<Body> {
    let mut builder = Response::builder()
        .status(status)
        .header("content-type", content_type)
        .header("cache-control", "no-store");
    if let Some(cookie) = cookie.and_then(|value| HeaderValue::from_str(&value).ok()) {
        builder = builder.header(SET_COOKIE, cookie);
    }
    match builder.body(body) {
        Ok(response) => response,
        Err(_) => empty_response(StatusCode::INTERNAL_SERVER_ERROR, None),
    }
}

fn redirect_response(location: &str, cookie: Option<String>) -> Response<Body> {
    let mut builder = Response::builder()
        .status(StatusCode::PERMANENT_REDIRECT)
        .header(LOCATION, location)
        .header("cache-control", "no-store");
    if let Some(cookie) = cookie.and_then(|value| HeaderValue::from_str(&value).ok()) {
        builder = builder.header(SET_COOKIE, cookie);
    }
    match builder.body(Body::empty()) {
        Ok(response) => response,
        Err(_) => empty_response(StatusCode::INTERNAL_SERVER_ERROR, None),
    }
}

fn empty_response(status: StatusCode, cookie: Option<String>) -> Response<Body> {
    let mut builder = Response::builder()
        .status(status)
        .header("cache-control", "no-store");
    if let Some(cookie) = cookie.and_then(|value| HeaderValue::from_str(&value).ok()) {
        builder = builder.header(SET_COOKIE, cookie);
    }
    match builder.body(Body::empty()) {
        Ok(response) => response,
        Err(_) => {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = status;
            response
        }
    }
}

fn route_values(routes: &RouteConfig) -> Vec<serde_json::Value> {
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
    .filter_map(|(name, route)| route.map(|route| json!({ "name": name, "path": route })))
    .collect()
}

fn target_value(target: &TargetSummary) -> serde_json::Value {
    json!({
        "name": &target.name,
        "kind": target.kind.as_str(),
        "model": &target.model,
        "backend": target.backend.as_ref().map(|backend| {
            json!({
                "selected": &backend.selected,
                "requested": backend.requested.as_str(),
                "reason": &backend.reason,
                "gpuOffloadExpected": backend.gpu_offload_expected,
            })
        }),
        "providerBaseUrl": &target.provider_base_url,
    })
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

fn clean_asset_path(path: &str) -> Option<PathBuf> {
    let mut clean = PathBuf::new();
    for component in Path::new(path.trim_start_matches('/')).components() {
        match component {
            Component::Normal(part) => clean.push(part),
            Component::CurDir => {}
            Component::ParentDir | Component::Prefix(_) | Component::RootDir => return None,
        }
    }
    if clean.as_os_str().is_empty() {
        clean.push("index.html");
    }
    Some(clean)
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("js") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("json") => "application/json",
        Some("ico") => "image/x-icon",
        Some("map") => "application/json",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

fn admin_base_path(route: &str) -> String {
    let route = route.trim_end_matches('/');
    if route.is_empty() {
        "/".to_string()
    } else {
        route.to_string()
    }
}

fn admin_base_with_slash(route: &str) -> String {
    if route == "/" {
        "/".to_string()
    } else {
        format!("{}/", route.trim_end_matches('/'))
    }
}

fn join_admin_route(base: &str, path: &str) -> String {
    let base = admin_base_path(base);
    if base == "/" {
        format!("/{path}")
    } else {
        format!("{base}/{path}")
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
