use crate::client::{SippError, EndpointError};

#[test]
fn gateway_endpoint_errors_remain_structured() {
    let mut endpoint = EndpointError::new("protocol", "invalid response");
    endpoint.status = Some(502);
    endpoint.code = Some("bad_upstream".to_string());
    let error = SippError::Endpoint(endpoint);
    assert_eq!(
        error.to_string(),
        "endpoint error (protocol): invalid response"
    );
}
