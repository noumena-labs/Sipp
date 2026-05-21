mod init;
mod observability;

pub(crate) use init::ensure_backend_initialized;
pub use init::set_llama_log_quiet;
pub use observability::backend_observability_json;
