use std::sync::Once;

static INIT_BACKEND: Once = Once::new();

pub(crate) fn ensure_backend_initialized() {
    INIT_BACKEND.call_once(|| {
        crate::native_bridge::backend_init();
        crate::native_bridge::backend_load_all();
    });
}

pub fn set_llama_log_quiet(quiet: bool) {
    crate::native_bridge::set_llama_log_quiet(quiet);
}
