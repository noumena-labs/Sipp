//! Tests the WASM lifecycle-service handle registry.

use super::*;

#[test]
fn lifecycle_handle_owns_catalog_state_without_an_engine() {
    let created = model_service_create_json("{}");
    let created: Value = serde_json::from_str(&created).expect("create response");
    let service = created["value"]["handle"].as_u64().expect("service handle") as usize;

    let listed = model_service_list_json(service);
    let listed: Value = serde_json::from_str(&listed).expect("list response");

    assert_eq!(listed["ok"], true);
    assert_eq!(listed["value"], serde_json::json!([]));
    assert_eq!(model_service_close(service), 1);
}
