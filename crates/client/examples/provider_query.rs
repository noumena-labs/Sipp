mod provider_common;

use cogentlm_client::{CogentQueryRequest, CogentTextOptions, CogentTextResponse, EndpointRef};
use futures::executor::block_on;

fn main() -> provider_common::ExampleResult<()> {
    block_on(async {
        let args = provider_common::args("Write one sentence about provider inference.")?;
        let mut client = cogentlm_client::CogentClient::new();
        client.add_provider_model(
            "openai",
            args.model.clone(),
            provider_common::openai_provider()?,
            cogentlm_client::ProviderExecutor::new()?,
        )?;
        let response = client
            .query(CogentQueryRequest {
                endpoint: Some(provider_endpoint(&args.model)),
                prompt: args.input,
                options: text_options(),
                ..Default::default()
            })
            .await?;
        print_text(response);
        Ok(())
    })
}

fn provider_endpoint(model: &str) -> EndpointRef {
    EndpointRef::ProviderModel {
        provider: "openai".to_string(),
        model: model.to_string(),
    }
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
