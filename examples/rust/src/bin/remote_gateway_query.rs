use cogentlm::{CogentClient, CogentQueryRequest};
use cogentlm_rust_examples::remote_common;
use futures::executor::block_on;

fn main() -> remote_common::ExampleResult<()> {
    block_on(async {
        let args = remote_common::args("Write one sentence about remote inference.")?;
        let mut client = CogentClient::new();
        let endpoint = client.add_remote(
            args.alias.clone(),
            remote_common::gateway_remote(args.alias)?,
        )?;
        let response = client
            .query(CogentQueryRequest {
                endpoint: Some(endpoint),
                prompt: args.input,
                options: remote_common::text_options(),
                ..Default::default()
            })
            .await?;
        remote_common::print_text(response);
        Ok(())
    })
}
