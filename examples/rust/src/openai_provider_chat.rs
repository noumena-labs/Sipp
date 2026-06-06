mod support;

use std::time::Duration;

use cogentlm::core::{ChatMessage, ChatRole};
use cogentlm_providers::{
    OpenAiAdapterConfig, ProviderChatRequest, ProviderGenerationOptions, ProviderOptions,
    ProviderTransport, SecretString,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> support::ExampleResult<()> {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Say hello from OpenAI in one sentence.".to_string()
    } else {
        prompt
    };

    // This example calls OpenAI directly through the provider adapter. It does
    // not start or use a CogentLM gateway, so the OpenAI key lives in this
    // process.
    let transport = ProviderTransport::openai(OpenAiAdapterConfig {
        api_key: SecretString::new(support::required_env("OPENAI_API_KEY")?),
        base_url: support::optional_env("OPENAI_BASE_URL"),
        timeout: Some(Duration::from_millis(
            support::env_parse("OPENAI_TIMEOUT_MS").unwrap_or(30_000),
        )),
    })?;

    let request = ProviderChatRequest {
        model: support::optional_env("OPENAI_MODEL").unwrap_or_else(|| "gpt-5-mini".to_string()),
        messages: vec![ChatMessage::new(ChatRole::User, prompt)],
        options: provider_options(),
        provider_options: ProviderOptions::new(),
    };

    if support::optional_env("COGENTLM_STREAM").as_deref() == Some("true") {
        let mut stream = transport.stream_chat(request).await?;
        while let Some(event) = stream.next().await {
            println!("event={:?}", event?);
        }
    } else {
        let response = transport.chat(request).await?;
        println!("provider={}", response.metadata.provider.as_str());
        println!("model={}", response.metadata.model);
        println!("text={}", response.result.text.trim());
        if let Some(usage) = response.usage {
            println!(
                "usage=input:{:?} output:{:?} total:{:?}",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
            );
        }
    }

    Ok(())
}

fn provider_options() -> ProviderGenerationOptions {
    ProviderGenerationOptions {
        max_tokens: support::env_parse("COGENTLM_MAX_TOKENS").or(Some(support::DEFAULT_MAX_TOKENS)),
        temperature: support::env_parse("COGENTLM_TEMPERATURE")
            .or(Some(support::DEFAULT_TEMPERATURE)),
        top_p: support::env_parse("COGENTLM_TOP_P").or(Some(support::DEFAULT_TOP_P)),
        stop: Vec::new(),
    }
}
