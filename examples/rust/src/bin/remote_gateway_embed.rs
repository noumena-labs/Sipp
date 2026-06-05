use cogentlm::{CogentClient, CogentEmbedRequest};
use cogentlm_rust_examples::remote_common;
use futures::executor::block_on;

fn main() -> remote_common::ExampleResult<()> {
    block_on(async {
        let args = remote_common::args("CogentClient remote embedding smoke input.")?;
        let mut client = CogentClient::new();
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
        remote_common::print_embedding(response);
        Ok(())
    })
}
