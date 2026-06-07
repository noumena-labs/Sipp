use crate::{map, CogentTextOptions};

#[test]
fn local_text_options_map_shared_generation_fields() {
    let options = map::local_chat_options(
        CogentTextOptions {
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
