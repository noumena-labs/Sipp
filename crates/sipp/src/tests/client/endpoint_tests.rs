use crate::core::CapabilitySupport;

use crate::client::{EndpointCapabilities, EndpointRef};

#[test]
fn endpoint_reference_exposes_only_its_id() {
    let endpoint = EndpointRef::from_id("edge");
    assert_eq!(endpoint.id(), "edge");
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
