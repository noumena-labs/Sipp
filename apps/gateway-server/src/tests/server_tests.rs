use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{
    header::{COOKIE, SET_COOKIE},
    Request, StatusCode,
};
use cogentlm_client::{
    CogentClient, EndpointDescriptor, GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes,
    GatewayTimeoutPolicy,
};
use cogentlm_engine::engine::{GpuLayerConfig, NativeRuntimeConfig};
use cogentlm_engine::lifecycle::{BackendCapabilities, StatsMode};
use tower::ServiceExt;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::config::{
    local_backend_plan_with_capabilities, EndpointConfig, GatewayBackendPreference,
    GatewayServerConfig, GatewayServerRuntime, LoadedToken, RouteConfig, TargetKind, TargetSummary,
};
use crate::http::GatewayHttpService;
use crate::metrics::GatewayMetrics;

async fn service(base_url: String) -> GatewayHttpService {
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
    let mut client = CogentClient::new();
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
    )
    .expect("service")
}

#[test]
fn config_accepts_typed_custom_routes() {
    let source = r#"
        public_bind = "127.0.0.1:8080"
        management_bind = "127.0.0.1:9090"
        admin_password = "admin-password"

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
    assert_eq!(config.admin_password, "admin-password");
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
        admin_password = "admin-password"

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
        admin_password = "admin-password"

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
        admin_password = "admin-password"

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
    let config = GatewayServerConfig {
        routes: RouteConfig::default(),
        admin_password: "admin-password".to_string(),
        ..GatewayServerConfig::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn config_rejects_missing_admin_password() {
    let source = r#"
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
    let error = config.validate().expect_err("admin password is required");
    assert!(error.to_string().contains("admin_password"));
}

#[test]
fn config_rejects_duplicate_routes_on_the_same_listener() {
    let mut routes = RouteConfig::default();
    routes.chat = routes.query.clone();
    let config = GatewayServerConfig {
        routes,
        admin_password: "admin-password".to_string(),
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
        ..GatewayServerConfig::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn shipped_gateway_configs_match_the_new_schema() {
    for name in ["development.toml", "production.toml"] {
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

#[tokio::test]
async fn admin_dashboard_requires_password_sessions_and_hides_secrets() {
    let upstream = MockServer::start().await;
    let router = service(upstream.uri()).await.management_router();

    let login_page = router
        .clone()
        .oneshot(Request::get("/admin").body(Body::empty()).expect("request"))
        .await
        .expect("response");
    assert_eq!(login_page.status(), StatusCode::OK);
    let login_body = response_text(login_page).await;
    assert!(login_body.contains("Gateway Admin"));
    assert!(!login_body.contains("admin-password"));
    assert!(!login_body.contains("test-secret"));

    let wrong_password = router
        .clone()
        .oneshot(
            Request::post("/admin/login")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("password=wrong"))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(wrong_password.status(), StatusCode::UNAUTHORIZED);

    let login = router
        .clone()
        .oneshot(
            Request::post("/admin/login")
                .header("content-type", "application/x-www-form-urlencoded")
                .body(Body::from("password=admin-password"))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(login.status(), StatusCode::SEE_OTHER);
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
            Request::get("/admin")
                .header(COOKIE, cookie.as_str())
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(dashboard.status(), StatusCode::OK);
    let dashboard_body = response_text(dashboard).await;
    assert!(dashboard_body.contains("allowed"));
    assert!(dashboard_body.contains("Admin password: configured in TOML"));
    assert!(!dashboard_body.contains("admin-password"));
    assert!(!dashboard_body.contains("test-secret"));

    let logout = router
        .oneshot(
            Request::post("/admin/logout")
                .header(COOKIE, cookie)
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(logout.status(), StatusCode::SEE_OTHER);
    let cleared = logout
        .headers()
        .get(SET_COOKIE)
        .and_then(|value| value.to_str().ok())
        .expect("clear cookie");
    assert!(cleared.contains("Max-Age=0"));
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
