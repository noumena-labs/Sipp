mod local_common;

use cogentlm_client::{CogentChatRequest, CogentTextOptions, CogentTextResponse, LocalTextOptions};
use cogentlm_engine::engine::{ChatMessage, ChatRole};
use futures::executor::block_on;
use futures::StreamExt;

fn main() -> local_common::ExampleResult<()> {
    block_on(async {
        let args = local_common::args("Explain the CogentClient API in one sentence.")?;
        let client = local_common::load_client(args.model_path, false).await?;
        let run = client.chat(CogentChatRequest {
            messages: vec![
                ChatMessage::new(ChatRole::System, "Answer concisely."),
                ChatMessage::new(ChatRole::User, args.input),
            ],
            options: text_options(),
            local: LocalTextOptions {
                context_key: Some("rust-chat-smoke".to_string()),
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
        print_text(response);
        Ok(())
    })
}

fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: local_common::env_parse("COGENTLM_MAX_TOKENS"),
        temperature: local_common::env_parse("COGENTLM_TEMPERATURE"),
        top_p: local_common::env_parse("COGENTLM_TOP_P"),
        stop: Vec::new(),
    }
}

fn print_text(response: CogentTextResponse) {
    println!("endpoint={:?}", response.endpoint);
    println!("finish_reason={}", response.finish_reason.as_str());
    println!("text={}", response.text.trim());
    if let Some(stats) = response.local_stats {
        println!(
            "metrics=ttft_ms:{:?} decode_ms:{:.3} output_tokens:{} e2e_tps:{:?} decode_tps:{:?}",
            stats.ttft_ms,
            stats.decode_ms,
            stats.output_tokens,
            stats.e2e_tokens_per_second,
            stats.decode_tokens_per_second
        );
    }
}
