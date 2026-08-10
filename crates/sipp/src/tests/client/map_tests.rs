use crate::client::{
    map, SippError, SippListenRequest, SippTextOptions, DEFAULT_TRANSCRIPTION_MAX_TOKENS,
};

#[test]
fn local_text_options_map_shared_generation_fields() {
    let options = map::local_chat_options(
        SippTextOptions {
            max_tokens: Some(32),
            temperature: Some(0.5),
            top_p: Some(0.9),
            stop: vec!["stop".to_string()],
        },
        Default::default(),
    )
    .expect("local options");
    assert_eq!(options.max_tokens, 32);
}

#[test]
fn local_listen_request_maps_explicit_and_default_token_limits() {
    let explicit = map::local_listen_request(SippListenRequest::new([1]).max_tokens(64))
        .expect("explicit listen request");
    assert_eq!(explicit.max_tokens, 64);

    let default =
        map::local_listen_request(SippListenRequest::new([1])).expect("default listen request");
    assert_eq!(
        default.max_tokens,
        i32::try_from(DEFAULT_TRANSCRIPTION_MAX_TOKENS).expect("default fits i32")
    );
}

#[test]
fn local_listen_request_rejects_token_limits_above_native_range() {
    let error =
        map::local_listen_request(SippListenRequest::new([1]).max_tokens(i32::MAX as u32 + 1))
            .expect_err("oversized listen max_tokens");
    assert!(matches!(
        error,
        SippError::InvalidRequest(message) if message == "local max_tokens exceeds i32::MAX"
    ));
}
