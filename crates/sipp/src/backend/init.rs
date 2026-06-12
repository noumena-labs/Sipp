use std::sync::{
    atomic::{AtomicBool, Ordering},
    Once,
};

static INIT_BACKEND: Once = Once::new();
const DEFAULT_LLAMA_LOG_QUIET: bool = true;
static LLAMA_LOG_QUIET: AtomicBool = AtomicBool::new(DEFAULT_LLAMA_LOG_QUIET);

pub(crate) fn ensure_backend_initialized() {
    INIT_BACKEND.call_once(|| {
        crate::native_bridge::set_llama_log_quiet(LLAMA_LOG_QUIET.load(Ordering::Relaxed));
        crate::native_bridge::backend_init();
        crate::native_bridge::backend_load_all();
    });
}

pub fn set_llama_log_quiet(quiet: bool) {
    LLAMA_LOG_QUIET.store(quiet, Ordering::Relaxed);
    crate::native_bridge::set_llama_log_quiet(quiet);
}

#[cfg(test)]
pub(crate) fn default_llama_log_quiet_for_tests() -> bool {
    DEFAULT_LLAMA_LOG_QUIET
}

#[cfg(test)]
#[path = "../tests/backend/init_tests.rs"]
mod init_tests;
