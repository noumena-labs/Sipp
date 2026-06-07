use cogentlm_client::{
    EndpointCapabilities, EndpointDescriptor, EndpointRef, GatewayEndpointConfig,
};

#[test]
fn gateway_descriptor_is_registered_through_add_contract() {
    let endpoint = EndpointRef::gateway("service");
    assert_eq!(endpoint.kind(), "gateway");
    let descriptor = EndpointDescriptor::gateway(GatewayEndpointConfig {
        target: "local".to_string(),
        base_url: "http://127.0.0.1:8080".to_string(),
        routes: Default::default(),
        authentication: Default::default(),
        static_headers: Default::default(),
        timeouts: Default::default(),
        protocol_options: Default::default(),
    });
    assert!(matches!(descriptor, EndpointDescriptor::Gateway(_)));
    assert_eq!(
        EndpointCapabilities::unknown().query,
        cogentlm_core::CapabilitySupport::Unknown
    );
}
