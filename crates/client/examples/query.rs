mod local_common;

use cogentlm_client::{
    CogentQueryRequest, CogentTextOptions, CogentTextResponse, LocalTextOptions,
};
use futures::executor::block_on;

fn main() -> local_common::ExampleResult<()> {
    block_on(async {
        let args = local_common::args("Write one sentence about local inference.")?;
        let client = local_common::load_client(args.model_path, false).await?;
        let response = client
            .query(CogentQueryRequest {
                prompt: args.input,
                options: text_options(),
                local: LocalTextOptions {
                    context_key: Some("rust-query-smoke".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;
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
