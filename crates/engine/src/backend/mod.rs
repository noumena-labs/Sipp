mod init;
mod observability;
mod observability_json;

pub(crate) use init::ensure_backend_initialized;
pub use init::set_llama_log_quiet;
pub use observability::backend_observability_json;
pub(crate) use observability_json::*;
