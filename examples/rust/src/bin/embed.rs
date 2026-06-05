use cogentlm::{CogentEmbedRequest, LocalEmbedOptions};
use cogentlm_rust_examples::local_common;
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
        local_common::print_embedding(response);
        Ok(())
    })
}
