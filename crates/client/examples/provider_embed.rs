mod provider_common;

use cogentlm_client::{CogentEmbedRequest, CogentEmbeddingResponse, EndpointRef};
use futures::executor::block_on;

fn main() -> provider_common::ExampleResult<()> {
    block_on(async {
        let args = provider_common::args("CogentClient provider embedding smoke input.")?;
        let mut client = cogentlm_client::CogentClient::new();
        client.add_provider_model(
            "openai",
            args.model.clone(),
            provider_common::openai_provider()?,
            cogentlm_client::ProviderExecutor::new()?,
        )?;
        let response = client
            .embed(CogentEmbedRequest {
                endpoint: Some(provider_endpoint(&args.model)),
                input: args.input,
                ..Default::default()
            })
            .await?;
        print_embedding(response);
        Ok(())
    })
}

fn provider_endpoint(model: &str) -> EndpointRef {
    EndpointRef::ProviderModel {
        provider: "openai".to_string(),
        model: model.to_string(),
    }
}

fn print_embedding(response: CogentEmbeddingResponse) {
    let preview = response
        .values
        .iter()
        .take(8)
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!("endpoint={:?}", response.endpoint);
    println!("dimensions={}", response.values.len());
    println!("preview=[{preview}]");
}
