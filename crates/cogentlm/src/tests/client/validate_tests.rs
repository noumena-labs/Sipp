use serde_json::json;

use crate::client::{validate, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions};

#[test]
fn local_requests_reject_gateway_endpoint_options() {
    let mut request = CogentQueryRequest::default();
    request
        .endpoint_options
        .insert("trace".to_string(), json!(true));
    assert!(matches!(
        validate::local_query(&request),
        Err(crate::client::CogentError::InvalidRequest(message))
            if message == "endpoint_options are not valid for local endpoints"
    ));

    let mut embed = CogentEmbedRequest::default();
    embed
        .endpoint_options
        .insert("normalize".to_string(), json!(true));
    assert!(validate::local_embed(&embed).is_err());
}

#[test]
fn common_text_options_reject_invalid_numbers() {
    assert!(validate::common_text_options(&CogentTextOptions {
        max_tokens: Some(0),
        ..Default::default()
    })
    .is_err());
    assert!(validate::common_text_options(&CogentTextOptions {
        top_p: Some(1.1),
        ..Default::default()
    })
    .is_err());
}
