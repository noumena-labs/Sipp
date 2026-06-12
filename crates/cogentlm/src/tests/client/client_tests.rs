use futures::executor::block_on;

use crate::client::{
    CogentClient, CogentError, EndpointDescriptor, EndpointRef, GatewayAuthentication,
    GatewayEndpointConfig, GatewayRoutes, GatewayTimeoutPolicy,
};

#[test]
fn registers_gateway_endpoint_through_add() {
    let mut client = CogentClient::new();
    let endpoint = block_on(client.add("gateway", EndpointDescriptor::gateway(gateway_config())))
        .expect("gateway endpoint");

    assert_eq!(
        endpoint,
        EndpointRef::Gateway {
            id: "gateway".to_string()
        }
    );
    assert!(client.resolve(Some(&endpoint), "chat").is_ok());
}

#[test]
fn replacing_an_id_keeps_single_registered_endpoint() {
    let mut client = CogentClient::new();
    let first = block_on(client.add("service", EndpointDescriptor::gateway(gateway_config())))
        .expect("first endpoint");
    let second = block_on(client.add("service", EndpointDescriptor::gateway(gateway_config())))
        .expect("replacement endpoint");

    assert_eq!(
        first,
        EndpointRef::Gateway {
            id: "service".to_string()
        }
    );
    assert_eq!(
        second,
        EndpointRef::Gateway {
            id: "service".to_string()
        }
    );
    assert!(client.resolve(Some(&second), "query").is_ok());
}

#[test]
fn gateway_endpoints_are_never_selected_implicitly() {
    let mut client = CogentClient::new();
    block_on(client.add("gateway", EndpointDescriptor::gateway(gateway_config())))
        .expect("gateway endpoint");

    assert!(matches!(
        client.resolve(None, "query"),
        Err(CogentError::NoSupportedEndpoint { operation: "query" })
    ));
}

fn gateway_config() -> GatewayEndpointConfig {
    GatewayEndpointConfig {
        target: "local".to_string(),
        base_url: "http://127.0.0.1:8080".to_string(),
        routes: GatewayRoutes::default(),
        authentication: GatewayAuthentication::None,
        static_headers: Default::default(),
        timeouts: GatewayTimeoutPolicy::default(),
        protocol_options: Default::default(),
    }
}
