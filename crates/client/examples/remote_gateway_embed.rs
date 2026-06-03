mod remote_common;

use cogentlm_client::{CogentEmbedRequest, CogentEmbeddingResponse};
use futures::executor::block_on;

fn main() -> remote_common::ExampleResult<()> {
    block_on(async {
        let args = remote_common::args("CogentClient remote embedding smoke input.")?;
        let mut client = cogentlm_client::CogentClient::new();
        let endpoint = client.add_remote(
            args.alias.clone(),
            remote_common::gateway_remote(args.alias)?,
        )?;
        let response = client
            .embed(CogentEmbedRequest {
                endpoint: Some(endpoint),
                input: args.input,
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
    println!("preview=[{preview}]");
}
