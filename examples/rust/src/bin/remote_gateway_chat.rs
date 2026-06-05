use cogentlm::engine::{ChatMessage, ChatRole};
use cogentlm::{CogentChatRequest, CogentClient};
use cogentlm_rust_examples::remote_common;
use futures::executor::block_on;
use futures::StreamExt;

fn main() -> remote_common::ExampleResult<()> {
    block_on(async {
        let args = remote_common::args("Explain remote inference in one sentence.")?;
        let mut client = CogentClient::new();
        let endpoint = client.add_remote(
            args.alias.clone(),
            remote_common::gateway_remote(args.alias)?,
        )?;
        let run = client.chat(CogentChatRequest {
            endpoint: Some(endpoint),
            messages: vec![
                ChatMessage::new(ChatRole::System, "Answer concisely."),
                ChatMessage::new(ChatRole::User, args.input),
            ],
            options: remote_common::text_options(),
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
        remote_common::print_text(response);
        Ok(())
    })
}
