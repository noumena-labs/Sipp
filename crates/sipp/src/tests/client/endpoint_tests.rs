use crate::core::CapabilitySupport;

use crate::client::{EndpointCapabilities, EndpointRef};

#[test]
fn gateway_endpoint_has_closed_kind() {
    let endpoint = EndpointRef::gateway("edge");
    assert_eq!(endpoint.id(), "edge");
    assert_eq!(endpoint.kind(), "gateway");
}

#[test]
fn unknown_capabilities_defer_to_endpoint_execution() {
    let capabilities = EndpointCapabilities::unknown();
    assert_eq!(
        capabilities.for_operation("query"),
        CapabilitySupport::Unknown
    );
    assert_eq!(
        capabilities.for_operation("unknown"),
        CapabilitySupport::Unsupported
    );
}
