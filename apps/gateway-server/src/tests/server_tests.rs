//! Tests for gateway-server configuration, routing, admin APIs, and
//! process-local runtime controls using model-free upstream fixtures.

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body};
use axum::extract::ConnectInfo;
use axum::http::{
    header::{COOKIE, SET_COOKIE},
    Request, StatusCode,
};
use sipp::engine::{GpuLayerConfig, NativeRuntimeConfig};
use sipp::lifecycle::{BackendCapabilities, StatsMode};
use sipp::{
    SippClient, EndpointDescriptor, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes,
    GatewayTimeoutPolicy,
};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::{
    local_backend_plan_with_capabilities, ClientIpConfig, ClientIpSource, EndpointConfig,
    GatewayBackendPreference, GatewayServerConfig, GatewayServerRuntime, LoadedToken,
    RateLimitConfig, RouteConfig, SecurityConfig, TargetKind, TargetSummary,
};
use crate::http::GatewayHttpService;
use crate::metrics::GatewayMetrics;

async fn service(base_url: String) -> GatewayHttpService {
    service_with_security(base_url, test_security_config()).await
}

async fn service_with_security(base_url: String, security: SecurityConfig) -> GatewayHttpService {
    let routes = RouteConfig {
        query: "/generate".to_string(),
        chat: "/conversation".to_string(),
        embed: "/vectorize".to_string(),
        index: Some("/about".to_string()),
        health: Some("/live".to_string()),
        readiness: Some("/ready".to_string()),
        metrics: Some("/telemetry".to_string()),
        admin: Some("/admin".to_string()),
    };
    let mut client = SippClient::new();
    let endpoint = client
        .add(
            "gateway-upstream",
            EndpointDescriptor::gateway(GatewayEndpointConfig {
                target: "upstream".to_string(),
                base_url: base_url.clone(),
                routes: GatewayRoutes::default(),
                authentication: GatewayAuthentication::None,
                static_headers: Default::default(),
                timeouts: GatewayTimeoutPolicy::default(),
                protocol_options: Default::default(),
            }),
        )
        .await
        .expect("gateway endpoint");
    let runtime = GatewayServerRuntime {
        client: Arc::new(client),
        targets: Arc::new(BTreeMap::from([("allowed".to_string(), endpoint)])),
        target_summaries: Arc::new(vec![TargetSummary {
            name: "allowed".to_string(),
            kind: TargetKind::OpenAiCompatible,
            model: "upstream".to_string(),
            backend: None,
            provider_base_url: Some(base_url),
        }]),
    };

    GatewayHttpService::new(
        runtime,
        routes,
        vec![LoadedToken {
            secret: "test-secret".to_string(),
            caller: "test-client".to_string(),
            targets: BTreeSet::from(["allowed".to_string()]),
        }],
        "admin-password".to_string(),
        Arc::new(GatewayMetrics::new()),
        1024,
        &[],
        None,
        security,
        admin_asset_dir(),
    )
    .expect("service")
}

fn test_security_config() -> SecurityConfig {
    SecurityConfig {
        client_ip: ClientIpConfig {
            source: ClientIpSource::Peer,
            trusted_proxy_cidrs: Vec::new(),
        },
        rate_limit: RateLimitConfig {
            enabled: false,
            requests_per_minute: 60,
            burst: 60,
        },
    }
}

fn test_gateway_config() -> GatewayServerConfig {
    GatewayServerConfig {
        public_bind: "127.0.0.1:8080".parse().expect("public bind"),
        management_bind: "127.0.0.1:9090".parse().expect("management bind"),
        max_request_bytes: 1024,
        allowed_origins: Vec::new(),
        admin_password_env: "GATEWAY_ADMIN_PASSWORD".to_string(),
        routes: RouteConfig::default(),
        tokens: vec![crate::config::TokenConfig {
            env: "GATEWAY_TEST_TOKEN".to_string(),
            caller: "developer".to_string(),
            targets: vec!["local".to_string()],
        }],
        targets: vec![crate::config::TargetConfig {
            name: "local".to_string(),
            endpoint: EndpointConfig::Local {
                model: "model.gguf".into(),
                backend: GatewayBackendPreference::Auto,
                stats: StatsMode::Basic,
                runtime: NativeRuntimeConfig::default(),
            },
        }],
        max_concurrent_requests: None,
        security: test_security_config(),
    }
}

#[test]
fn config_accepts_typed_custom_routes() {
    let source = r#"
        public_bind = "127.0.0.1:8080"
        management_bind = "127.0.0.1:9090"
        admin_password_env = "GATEWAY_ADMIN_PASSWORD"

        [security.client_ip]
        source = "peer"
        trusted_proxy_cidrs = []

        [security.rate_limit]
        enabled = false
        requests_per_minute = 60
        burst = 60

        [routes]
        query = "/generate"
        chat = "/conversation"
        embed = "/vectorize"
        index = "/about"
        health = "/live"
        readiness = "/ready"
        metrics = "/telemetry"
        admin = "/admin"

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
    "#;

    let config: GatewayServerConfig = toml::from_str(source).expect("config");
    config.validate().expect("valid config");
    assert_eq!(config.routes.query, "/generate");
    assert_eq!(config.routes.metrics.as_deref(), Some("/telemetry"));
    assert_eq!(config.routes.admin.as_deref(), Some("/admin"));
    assert_eq!(config.admin_password_env, "GATEWAY_ADMIN_PASSWORD");
    match &config.targets[0].endpoint {
        EndpointConfig::Local { backend, stats, .. } => {
            assert_eq!(*backend, GatewayBackendPreference::Auto);
            assert_eq!(*stats, StatsMode::Basic);
        }
        _ => panic!("expected local target"),
    }
}

#[test]
fn local_target_accepts_backend_and_stats_overrides() {
    let source = r#"
        admin_password_env = "GATEWAY_ADMIN_PASSWORD"

        [security.client_ip]
        source = "peer"
        trusted_proxy_cidrs = []

        [security.rate_limit]
        enabled = false
        requests_per_minute = 60
        burst = 60

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
        backend = "vulkan"
        stats = "profile"
    "#;

    let config: GatewayServerConfig = toml::from_str(source).expect("config");
    config.validate().expect("valid config");
    match &config.targets[0].endpoint {
        EndpointConfig::Local { backend, stats, .. } => {
            assert_eq!(*backend, GatewayBackendPreference::Vulkan);
            assert_eq!(*stats, StatsMode::Profile);
        }
        _ => panic!("expected local target"),
    }
}

#[test]
fn local_target_rejects_unsupported_backend_names() {
    let source = r#"
        admin_password_env = "GATEWAY_ADMIN_PASSWORD"

        [security.client_ip]
        source = "peer"
        trusted_proxy_cidrs = []

        [security.rate_limit]
        enabled = false
        requests_per_minute = 60
        burst = 60

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
        backend = "webgpu"
    "#;

    let error = toml::from_str::<GatewayServerConfig>(source)
        .err()
        .expect("unsupported backend");
    assert!(error.to_string().contains("unknown variant"));
}

#[test]
fn provider_targets_keep_server_side_secret_envs() {
    let source = r#"
        admin_password_env = "GATEWAY_ADMIN_PASSWORD"

        [security.client_ip]
        source = "peer"
        trusted_proxy_cidrs = []

        [security.rate_limit]
        enabled = false
        requests_per_minute = 60
        burst = 60

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["upstream"]

        [[targets]]
        name = "upstream"
        type = "openai_compatible"
        model = "served-model"
        base_url = "https://provider.example/v1"
        token_env = "PROVIDER_TOKEN"
    "#;

    let config: GatewayServerConfig = toml::from_str(source).expect("config");
    config.validate().expect("valid config");
    match &config.targets[0].endpoint {
        EndpointConfig::OpenaiCompatible {
            model,
            base_url,
            token_env,
            ..
        } => {
            assert_eq!(model, "served-model");
            assert_eq!(base_url, "https://provider.example/v1");
            assert_eq!(token_env, "PROVIDER_TOKEN");
        }
        _ => panic!("expected provider target"),
    }
}

#[test]
fn local_auto_backend_selects_best_available_backend() {
    let plan = local_backend_plan_with_capabilities(
        GatewayBackendPreference::Auto,
        StatsMode::Basic,
        NativeRuntimeConfig::default(),
        &backend_capabilities(&["cpu", "vulkan", "cuda"], &["cpu", "vulkan", "cuda"], true),
    )
    .expect("backend plan");

    assert_eq!(plan.selection.selected, "cuda");
    assert_eq!(plan.selection.requested.as_str(), "auto");
    assert!(plan.selection.gpu_offload_expected);
}

#[test]
fn local_cpu_backend_disables_gpu_offload() {
    let mut runtime = NativeRuntimeConfig::default();
    runtime.placement.devices = vec!["cuda0".to_string()];
    runtime.placement.gpu_layers = GpuLayerConfig::All;
    runtime.context.offload_kqv = true;
    runtime.context.op_offload = true;

    let plan = local_backend_plan_with_capabilities(
        GatewayBackendPreference::Cpu,
        StatsMode::Off,
        runtime,
        &backend_capabilities(&["cpu", "cuda"], &["cpu", "cuda"], true),
    )
    .expect("backend plan");

    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
    assert!(plan.config.placement.devices.is_empty());
    assert!(!plan.config.context.offload_kqv);
    assert!(!plan.config.context.op_offload);
    assert!(!plan.config.observability.runtime_metrics);
    assert!(!plan.config.observability.backend_profiling);
}

#[test]
fn explicit_unavailable_backend_returns_an_error() {
    let error = local_backend_plan_with_capabilities(
        GatewayBackendPreference::Cuda,
        StatsMode::Basic,
        NativeRuntimeConfig::default(),
        &backend_capabilities(&["cpu"], &["cpu"], false),
    )
    .expect_err("backend unavailable");

    assert!(error.to_string().contains("requested backend cuda"));
}

#[test]
fn config_rejects_missing_application_policy() {
    let mut config = test_gateway_config();
    config.tokens.clear();
    config.targets.clear();

    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_missing_security_section() {
    let source = r#"
        admin_password_env = "GATEWAY_ADMIN_PASSWORD"

        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
    "#;

    let error = match toml::from_str::<GatewayServerConfig>(source) {
        Ok(_) => panic!("security section is required"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("security"));
}

#[test]
fn config_rejects_missing_admin_password_env() {
    let source = r#"
        [[tokens]]
        env = "GATEWAY_TEST_TOKEN"
        caller = "developer"
        targets = ["local"]

        [security.client_ip]
        source = "peer"
        trusted_proxy_cidrs = []

        [security.rate_limit]
        enabled = false
        requests_per_minute = 60
        burst = 60

        [[targets]]
        name = "local"
        type = "local"
        model = "model.gguf"
    "#;

    let config: GatewayServerConfig = toml::from_str(source).expect("config");
    let error = config
        .validate()
        .expect_err("admin password env is required");
    assert!(error.to_string().contains("admin_password_env"));
}

#[test]
fn config_rejects_invalid_admin_password_env_name() {
    let mut config = test_gateway_config();
    config.admin_password_env = "SIPP GATEWAY ADMIN PASSWORD".to_string();

    let error = config
        .validate()
        .expect_err("admin password env name is invalid");
    assert!(error.to_string().contains("environment variable"));
}

#[test]
fn load_admin_password_reads_non_empty_secret_env() {
    let mut config = test_gateway_config();
    let env_name = unique_env_name("GATEWAY_ADMIN_PASSWORD");
    config.admin_password_env = env_name.clone();

    std::env::set_var(&env_name, "admin-secret");
    assert_eq!(
        config.load_admin_password().expect("admin password"),
        "admin-secret"
    );

    std::env::set_var(&env_name, " ");
    let error = config
        .load_admin_password()
        .expect_err("blank admin password is rejected");
    assert!(error.to_string().contains("must not be empty"));
    std::env::remove_var(env_name);
}

#[test]
fn config_rejects_duplicate_routes_on_the_same_listener() {
    let mut config = test_gateway_config();
    config.routes.chat = config.routes.query.clone();

    assert!(config.validate().is_err());
}

#[test]
fn shipped_gateway_configs_match_the_new_schema() {
    for name in [
        "local.toml.example",
        "development.toml.example",
        "production.toml.example",
        "provider-only.toml.example",
        "hybrid.toml.example",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("config")
            .join(name);
        GatewayServerConfig::from_path(&path).expect(name);
    }
}

#[tokio::test]
async fn public_routes_require_authentication_apply_policy_and_call_client() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "upstream-response",
            "model": "upstream",
            "text": "hello",
            "finish_reason": "stop"
        })))
        .mount(&upstream)
        .await;

    let router = service(upstream.uri()).await.public_router();
    let missing = router
        .clone()
        .oneshot(
            Request::post("/generate")
                .extension(ConnectInfo(test_peer()))
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"allowed","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(missing.status(), StatusCode::UNAUTHORIZED);

    let forbidden = router
        .clone()
        .oneshot(
            Request::post("/generate")
                .extension(ConnectInfo(test_peer()))
                .header("authorization", "Bearer test-secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"blocked","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(forbidden.status(), StatusCode::FORBIDDEN);

    let allowed = router
        .oneshot(
            Request::post("/generate")
                .extension(ConnectInfo(test_peer()))
                .header("authorization", "Bearer test-secret")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"model":"allowed","prompt":"hello"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(allowed.status(), StatusCode::OK);
    let body = to_bytes(allowed.into_body(), 4096).await.expect("body");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(body["text"], "hello");
    assert_eq!(body["model"], "allowed");
}

#[tokio::test]
async fn security_controls_are_in_memory_and_reset_with_service_state() {
    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/query"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "id": "upstream-response",
            "model": "upstream",
            "text": "hello",
            "finish_reason": "stop"
        })))
        .mount(&upstream)
        .await;
    let security = SecurityConfig {
        client_ip: ClientIpConfig {
            source: ClientIpSource::XRealIp,
            trusted_proxy_cidrs: vec!["0.0.0.0/0".to_string()],
        },
        rate_limit: RateLimitConfig {
            enabled: true,
            requests_per_minute: 1,
            burst: 1,
        },
    };
    let service = service_with_security(upstream.uri(), security.clone()).await;
    let public = service.public_router();

    let allowed = public
        .clone()
        .oneshot(authorized_query("127.0.0.8"))
        .await
        .expect("response");
    assert_eq!(allowed.status(), StatusCode::OK);

    let limited = public
        .clone()
        .oneshot(authorized_query("127.0.0.8"))
        .await
        .expect("response");
    assert_eq!(limited.status(), StatusCode::TOO_MANY_REQUESTS);

    let management = service.management_router();
    let login = management
        .clone()
        .oneshot(
            Request::post("/admin/api/session")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"admin-password"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    let cookie = login
        .headers()
        .get(SET_COOKIE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .expect("session cookie")
        .to_string();
    let session = management
        .clone()
        .oneshot(
            Request::get("/admin/api/session")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    let csrf = response_json(session).await["csrfToken"]
        .as_str()
        .expect("csrf token")
        .to_string();
    let block = management
        .oneshot(
            Request::post("/admin/api/security/blocklist/127.0.0.9")
                .header(COOKIE, cookie)
                .header("x-sipp-admin-csrf", csrf)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(block.status(), StatusCode::OK);

    let blocked = public
        .oneshot(authorized_query("127.0.0.9"))
        .await
        .expect("response");
    assert_eq!(blocked.status(), StatusCode::FORBIDDEN);

    let restarted = service_with_security(upstream.uri(), security)
        .await
        .public_router();
    let reset = restarted
        .oneshot(authorized_query("127.0.0.9"))
        .await
        .expect("response");
    assert_eq!(reset.status(), StatusCode::OK);
}

#[tokio::test]
async fn management_routes_are_application_owned() {
    let upstream = MockServer::start().await;
    let router = service(upstream.uri()).await.management_router();

    for (route, expected) in [("/live", "ok"), ("/ready", "ready")] {
        let response = router
            .clone()
            .oneshot(Request::get(route).body(Body::empty()).expect("request"))
            .await
            .expect("response");
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 4096).await.expect("body");
        assert_eq!(body, expected);
    }

    let metrics = router
        .oneshot(
            Request::get("/telemetry")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(metrics.status(), StatusCode::OK);
}

fn authorized_query(client: &str) -> Request<Body> {
    Request::post("/generate")
        .extension(ConnectInfo(test_peer()))
        .header("authorization", "Bearer test-secret")
        .header("content-type", "application/json")
        .header("x-real-ip", client)
        .body(Body::from(r#"{"model":"allowed","prompt":"hello"}"#))
        .expect("request")
}

fn test_peer() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 50000))
}

#[tokio::test]
async fn admin_dashboard_requires_password_sessions_and_hides_secrets() {
    let upstream = MockServer::start().await;
    let router = service(upstream.uri()).await.management_router();

    let login_redirect = router
        .clone()
        .oneshot(Request::get("/admin").body(Body::empty()).expect("request"))
        .await
        .expect("response");
    assert_eq!(login_redirect.status(), StatusCode::PERMANENT_REDIRECT);

    let spa = router
        .clone()
        .oneshot(
            Request::get("/admin/")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(spa.status(), StatusCode::OK);
    let spa_body = response_text(spa).await;
    assert!(spa_body.contains("Sipp Gateway Admin"));
    assert!(!spa_body.contains("admin-password"));
    assert!(!spa_body.contains("test-secret"));

    let unauthorized_session = router
        .clone()
        .oneshot(
            Request::get("/admin/api/session")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(unauthorized_session.status(), StatusCode::UNAUTHORIZED);

    let wrong_password = router
        .clone()
        .oneshot(
            Request::post("/admin/api/session")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"wrong"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(wrong_password.status(), StatusCode::UNAUTHORIZED);

    let login = router
        .clone()
        .oneshot(
            Request::post("/admin/api/session")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"password":"admin-password"}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(login.status(), StatusCode::OK);
    let cookie = login
        .headers()
        .get(SET_COOKIE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .expect("session cookie")
        .to_string();

    let dashboard = router
        .clone()
        .oneshot(
            Request::get("/admin/")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(dashboard.status(), StatusCode::OK);
    let dashboard_body = response_text(dashboard).await;
    assert!(dashboard_body.contains("Sipp Gateway Admin"));
    assert!(!dashboard_body.contains("<base"));
    assert!(!dashboard_body.contains("admin-password"));
    assert!(!dashboard_body.contains("test-secret"));

    let session = router
        .clone()
        .oneshot(
            Request::get("/admin/api/session")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(session.status(), StatusCode::OK);
    let session = response_json(session).await;
    let csrf = session["csrfToken"]
        .as_str()
        .expect("csrf token")
        .to_string();

    let targets = router
        .clone()
        .oneshot(
            Request::get("/admin/api/targets")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(targets.status(), StatusCode::OK);
    let targets_body = response_text(targets).await;
    assert!(targets_body.contains("allowed"));
    assert!(!targets_body.contains("admin-password"));
    assert!(!targets_body.contains("test-secret"));

    let blocked_without_csrf = router
        .clone()
        .oneshot(
            Request::post("/admin/api/security/blocklist/127.0.0.9")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(blocked_without_csrf.status(), StatusCode::FORBIDDEN);

    let block = router
        .clone()
        .oneshot(
            Request::post("/admin/api/security/blocklist/127.0.0.9")
                .header(COOKIE, cookie.as_str())
                .header("x-sipp-admin-csrf", csrf.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(block.status(), StatusCode::OK);
    assert!(response_text(block).await.contains("127.0.0.9"));

    let concurrency = router
        .clone()
        .oneshot(
            Request::put("/admin/api/controls/concurrency")
                .header(COOKIE, cookie.as_str())
                .header("x-sipp-admin-csrf", csrf.as_str())
                .header("content-type", "application/json")
                .body(Body::from(r#"{"limit":1}"#))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(concurrency.status(), StatusCode::OK);

    let logout = router
        .oneshot(
            Request::delete("/admin/api/session")
                .header(COOKIE, cookie)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(logout.status(), StatusCode::OK);
    let cleared = logout
        .headers()
        .get(SET_COOKIE)
        .and_then(|value| value.to_str().ok())
        .expect("clear cookie");
    assert!(cleared.contains("Max-Age=0"));
}

fn unique_env_name(prefix: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    format!("{prefix}_{}_{}", std::process::id(), now)
}

fn admin_asset_dir() -> std::path::PathBuf {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    let dir = std::env::temp_dir()
        .join("sipp-gateway-admin-test")
        .join(format!("{}-{now}", std::process::id()));
    std::fs::create_dir_all(dir.join("assets")).expect("asset dir");
    std::fs::write(
        dir.join("index.html"),
        r#"<!doctype html><title>Sipp Gateway Admin</title><div id="root"></div>"#,
    )
    .expect("index");
    std::fs::write(dir.join("assets").join("app.js"), "export {};").expect("asset");
    dir
}

fn backend_capabilities(
    compiled: &[&str],
    available: &[&str],
    gpu_offload_supported: bool,
) -> BackendCapabilities {
    BackendCapabilities {
        compiled: compiled.iter().map(|value| (*value).to_string()).collect(),
        available: available.iter().map(|value| (*value).to_string()).collect(),
        gpu_offload_supported,
    }
}

async fn response_text(response: axum::http::Response<Body>) -> String {
    let bytes = to_bytes(response.into_body(), 16 * 1024)
        .await
        .expect("body");
    String::from_utf8(bytes.to_vec()).expect("utf8 body")
}

async fn response_json(response: axum::http::Response<Body>) -> serde_json::Value {
    let text = response_text(response).await;
    serde_json::from_str(&text).expect("json body")
}
