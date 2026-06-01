mod local_common;

use cogentlm_client::{CogentEmbedRequest, CogentEmbeddingResponse, LocalEmbedOptions};
use futures::executor::block_on;

fn main() -> local_common::ExampleResult<()> {
    block_on(async {
        let args = local_common::args("CogentClient embedding smoke input.")?;
        let client = local_common::load_client(args.model_path, true).await?;
        let response = client
            .embed(CogentEmbedRequest {
                input: args.input,
                local: LocalEmbedOptions {
                    context_key: Some("rust-embed-smoke".to_string()),
                    normalize: Some(true),
                },
                ..Default::default()
            })
            .await?;
        print_embedding(response);
        Ok(())
    })
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
    println!("pooling={:?}", response.pooling);
    println!("normalized={:?}", response.normalized);
    println!("preview=[{preview}]");
}
