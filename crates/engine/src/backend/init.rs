use std::sync::Once;

static INIT_BACKEND: Once = Once::new();

pub(crate) fn ensure_backend_initialized() {
    INIT_BACKEND.call_once(|| unsafe {
        cogentlm_sys::llama_backend_init();
        cogentlm_sys::cogent_backend_load_all();
    });
}

pub fn set_llama_log_quiet(quiet: bool) {
    unsafe {
        cogentlm_sys::cogent_set_llama_log_quiet(quiet);
    }
}
