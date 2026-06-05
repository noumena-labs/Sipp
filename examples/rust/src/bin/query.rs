use cogentlm::{CogentQueryRequest, LocalTextOptions};
use cogentlm_rust_examples::local_common;
use futures::executor::block_on;

fn main() -> local_common::ExampleResult<()> {
    block_on(async {
        let args = local_common::args("Write one sentence about local inference.")?;
        let client = local_common::load_client(args.model_path, false).await?;
        let response = client
            .query(CogentQueryRequest {
                prompt: args.input,
                options: local_common::text_options(),
                local: LocalTextOptions {
                    context_key: Some("rust-query-smoke".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            })
            .await?;
        local_common::print_text(response);
        Ok(())
    })
}
