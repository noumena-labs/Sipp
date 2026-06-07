//! Tests the shipped container and Compose fixtures without invoking Docker or
//! requiring a model artifact.

#[test]
fn dockerfile_builds_cpu_service_and_runs_as_non_root() {
    let dockerfile = include_str!("../../Dockerfile");

    assert!(dockerfile.contains("cargo xtask build core"));
    assert!(dockerfile.contains("USER 10001:10001"));
    assert!(dockerfile.contains(r#"ENTRYPOINT ["cogentlm-gateway"]"#));
}

#[test]
fn compose_keeps_management_port_on_loopback_and_sets_shutdown_grace() {
    let compose = include_str!("../../compose.yaml");

    assert!(compose.contains(r#""127.0.0.1:9090:9090""#));
    assert!(compose.contains("stop_grace_period: 130s"));
    assert!(compose.contains("http://127.0.0.1:9090/readyz"));
    assert!(compose.contains("read_only: true"));
}
