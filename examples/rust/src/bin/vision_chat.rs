use std::fs;

use cogentlm::engine::{ChatMessage, ChatRole};
use cogentlm::{CogentChatRequest, LocalTextOptions};
use cogentlm_rust_examples::local_common;
use futures::executor::block_on;
use futures::StreamExt;

fn main() -> local_common::ExampleResult<()> {
    block_on(async {
        let args = local_common::vision_args("Describe this image in one sentence.")?;
        let image = fs::read(args.image_path)?;
        let client = local_common::load_client_with_projector(
            args.model_path,
            Some(args.projector_path),
            false,
        )
        .await?;
        let run = client.chat(CogentChatRequest {
            messages: vec![ChatMessage::new(ChatRole::User, args.input)],
            options: local_common::text_options(),
            local: LocalTextOptions {
                context_key: Some("rust-vision-chat-smoke".to_string()),
                media: vec![image],
                ..Default::default()
            },
            emit_tokens: true,
            ..Default::default()
        });
        let (mut tokens, response) = run.into_parts();
        let mut streamed = String::new();
        while let Some(batch) = tokens.next().await {
            print!("{}", batch.text);
            streamed.push_str(&batch.text);
        }
        println!();
        let response = response.await?;
        assert_eq!(streamed, response.text);
        local_common::print_text(response);
        Ok(())
    })
}
