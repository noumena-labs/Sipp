mod remote_common;

use cogentlm_client::{CogentQueryRequest, CogentTextOptions, CogentTextResponse};
use futures::executor::block_on;

fn main() -> remote_common::ExampleResult<()> {
    block_on(async {
        let args = remote_common::args("Write one sentence about remote inference.")?;
        let mut client = cogentlm_client::CogentClient::new();
        let endpoint = client.add_remote(
            args.alias.clone(),
            remote_common::gateway_remote(args.alias)?,
        )?;
        let response = client
            .query(CogentQueryRequest {
                endpoint: Some(endpoint),
                prompt: args.input,
                options: text_options(),
                ..Default::default()
            })
            .await?;
        print_text(response);
        Ok(())
    })
}

fn text_options() -> CogentTextOptions {
    CogentTextOptions {
        max_tokens: env_parse("COGENTLM_MAX_TOKENS"),
        temperature: env_parse("COGENTLM_TEMPERATURE"),
        top_p: env_parse("COGENTLM_TOP_P"),
        stop: Vec::new(),
    }
}

fn print_text(response: CogentTextResponse) {
    println!("endpoint={:?}", response.endpoint);
    println!("finish_reason={}", response.finish_reason.as_str());
    println!("text={}", response.text.trim());
}

fn env_parse<T>(name: &'static str) -> Option<T>
where
    T: std::str::FromStr,
{
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}
