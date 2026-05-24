use std::os::raw::{c_char, c_float, c_int, c_void};

pub const LLAMA_DEFAULT_SEED: u32 = 0xFFFF_FFFF;
pub const LLAMA_TOKEN_NULL: i32 = -1;
pub const LLAMA_STATE_SEQ_FLAGS_NONE: u32 = 0;
pub const LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY: u32 = 1;
pub const LLAMA_STATE_SEQ_FLAGS_ON_DEVICE: u32 = 2;

pub type llama_pos = i32;
pub type llama_token = i32;
pub type llama_seq_id = i32;
pub type llama_split_mode = c_int;
pub type llama_rope_scaling_type = c_int;
pub type llama_pooling_type = c_int;
pub type llama_attention_type = c_int;
pub type llama_flash_attn_type = c_int;
pub type ggml_type = c_int;
pub type llama_memory_t = *mut c_void;

pub const LLAMA_SPLIT_MODE_NONE: llama_split_mode = 0;
pub const LLAMA_SPLIT_MODE_LAYER: llama_split_mode = 1;
pub const LLAMA_SPLIT_MODE_ROW: llama_split_mode = 2;
pub const LLAMA_SPLIT_MODE_TENSOR: llama_split_mode = 3;

#[repr(C)]
pub struct llama_vocab {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_model {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_context {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_sampler {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct ggml_tensor {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct ggml_backend_device {
    _unused: [u8; 0],
}

pub type ggml_backend_dev_t = *mut ggml_backend_device;
pub type ggml_backend_buffer_type_t = *mut c_void;
pub type ggml_backend_sched_eval_callback =
    Option<unsafe extern "C" fn(t: *mut ggml_tensor, ask: bool, user_data: *mut c_void) -> bool>;
pub type ggml_abort_callback = Option<unsafe extern "C" fn(data: *mut c_void) -> bool>;
pub type llama_progress_callback =
    Option<unsafe extern "C" fn(progress: c_float, user_data: *mut c_void) -> bool>;

#[repr(C)]
pub struct llama_model_kv_override {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct llama_model_tensor_buft_override {
    pub pattern: *const c_char,
    pub buft: ggml_backend_buffer_type_t,
}

#[repr(C)]
pub struct llama_model_params {
    pub devices: *mut ggml_backend_dev_t,
    pub tensor_buft_overrides: *const llama_model_tensor_buft_override,
    pub n_gpu_layers: i32,
    pub split_mode: llama_split_mode,
    pub main_gpu: i32,
    pub tensor_split: *const c_float,
    pub progress_callback: llama_progress_callback,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const llama_model_kv_override,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

#[repr(C)]
pub struct llama_sampler_seq_config {
    pub seq_id: llama_seq_id,
    pub sampler: *mut llama_sampler,
}

#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub rope_scaling_type: llama_rope_scaling_type,
    pub pooling_type: llama_pooling_type,
    pub attention_type: llama_attention_type,
    pub flash_attn_type: llama_flash_attn_type,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: ggml_backend_sched_eval_callback,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: ggml_type,
    pub type_v: ggml_type,
    pub abort_callback: ggml_abort_callback,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut llama_sampler_seq_config,
    pub n_samplers: usize,
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

#[repr(C)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut llama_pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut llama_seq_id,
    pub logits: *mut i8,
}

#[repr(C)]
pub struct cogent_chat_templates {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_common_params {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_common_init {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_common_sampler {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_common_checkpoint {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_mtmd_context {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_mtmd_bitmap {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct cogent_mtmd_input_chunks {
    _unused: [u8; 0],
}

extern "C" {
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;
    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;
    pub fn llama_backend_init();
    pub fn llama_backend_free();
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_get_memory(ctx: *const llama_context) -> llama_memory_t;
    pub fn llama_model_n_ctx_train(model: *const llama_model) -> i32;
    pub fn llama_model_is_recurrent(model: *const llama_model) -> bool;
    pub fn llama_model_is_hybrid(model: *const llama_model) -> bool;
    pub fn llama_memory_seq_rm(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
    ) -> bool;
    pub fn llama_memory_seq_add(
        mem: llama_memory_t,
        seq_id: llama_seq_id,
        p0: llama_pos,
        p1: llama_pos,
        delta: llama_pos,
    );
    pub fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: llama_seq_id) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut u8,
        size: usize,
        seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_state_seq_set_data(
        ctx: *mut llama_context,
        src: *const u8,
        size: usize,
        dest_seq_id: llama_seq_id,
    ) -> usize;
    pub fn llama_n_ctx(ctx: *const llama_context) -> u32;
    pub fn llama_n_batch(ctx: *const llama_context) -> u32;
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> i32;
    pub fn llama_synchronize(ctx: *mut llama_context);
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;
    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_is_eog(vocab: *const llama_vocab, token: llama_token) -> bool;
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, smpl: *mut llama_sampler);
    pub fn llama_sampler_free(smpl: *mut llama_sampler);
    pub fn llama_sampler_reset(smpl: *mut llama_sampler);
    pub fn llama_sampler_clone(smpl: *const llama_sampler) -> *mut llama_sampler;
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_min_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(t: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_penalties(
        penalty_last_n: i32,
        penalty_repeat: c_float,
        penalty_freq: c_float,
        penalty_present: c_float,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_init_grammar(
        vocab: *const llama_vocab,
        grammar_str: *const c_char,
        grammar_root: *const c_char,
    ) -> *mut llama_sampler;
    pub fn llama_sampler_sample(
        smpl: *mut llama_sampler,
        ctx: *mut llama_context,
        idx: i32,
    ) -> llama_token;
    pub fn llama_sampler_accept(smpl: *mut llama_sampler, token: llama_token);
    pub fn llama_set_sampler(
        ctx: *mut llama_context,
        seq_id: llama_seq_id,
        smpl: *mut llama_sampler,
    ) -> bool;

    pub fn cogent_chat_templates_init(
        model: *const llama_model,
        chat_template_override: *const c_char,
    ) -> *mut cogent_chat_templates;
    pub fn cogent_chat_templates_free(templates: *mut cogent_chat_templates);
    pub fn cogent_chat_templates_source(templates: *const cogent_chat_templates) -> *mut c_char;
    pub fn cogent_apply_chat_template(
        templates: *const cogent_chat_templates,
        messages_json: *const c_char,
        add_assistant: bool,
    ) -> *mut c_char;
    pub fn cogent_free_string(value: *mut c_char);
    pub fn cogent_free_buffer(value: *mut c_void);
    pub fn cogent_common_params_parse_server(
        model_path: *const c_char,
        argc: i32,
        argv: *const *const c_char,
        error_out: *mut *mut c_char,
    ) -> *mut cogent_common_params;
    pub fn cogent_common_params_free(params: *mut cogent_common_params);
    pub fn cogent_common_init_from_params(
        params: *const cogent_common_params,
        error_out: *mut *mut c_char,
    ) -> *mut cogent_common_init;
    pub fn cogent_common_init_free(init: *mut cogent_common_init);
    pub fn cogent_common_init_model(init: *mut cogent_common_init) -> *mut llama_model;
    pub fn cogent_common_init_context(init: *mut cogent_common_init) -> *mut llama_context;
    pub fn cogent_common_init_vocab(init: *mut cogent_common_init) -> *const llama_vocab;
    pub fn cogent_common_init_n_parallel(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_n_batch(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_n_ubatch(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_n_ctx(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_n_embd_out(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_pooling_type(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_decoder_start_token(init: *const cogent_common_init) -> i32;
    pub fn cogent_common_init_model_has_encoder(init: *const cogent_common_init) -> bool;
    pub fn cogent_common_init_model_has_decoder(init: *const cogent_common_init) -> bool;
    pub fn cogent_common_init_model_has_chat_template(init: *const cogent_common_init) -> bool;
    pub fn cogent_common_init_kv_unified(init: *const cogent_common_init) -> bool;
    pub fn cogent_common_init_flash_attention(init: *const cogent_common_init) -> *mut c_char;
    pub fn cogent_common_init_cache_type_k(init: *const cogent_common_init) -> *mut c_char;
    pub fn cogent_common_init_cache_type_v(init: *const cogent_common_init) -> *mut c_char;
    pub fn cogent_common_sampler_init_from_json(
        init: *mut cogent_common_init,
        sampling_json: *const c_char,
        grammar: *const c_char,
        json_schema: *const c_char,
        error_out: *mut *mut c_char,
    ) -> *mut cogent_common_sampler;
    pub fn cogent_common_sampler_free(sampler: *mut cogent_common_sampler);
    pub fn cogent_common_sampler_raw(sampler: *mut cogent_common_sampler) -> *mut llama_sampler;
    pub fn cogent_common_sampler_backend_sampling(sampler: *const cogent_common_sampler) -> bool;
    pub fn cogent_common_sampler_print(sampler: *const cogent_common_sampler) -> *mut c_char;
    pub fn cogent_common_sampler_sample(
        sampler: *mut cogent_common_sampler,
        context: *mut llama_context,
        idx: c_int,
    ) -> llama_token;
    pub fn cogent_common_sampler_accept(
        sampler: *mut cogent_common_sampler,
        token: llama_token,
        is_generated: bool,
    ) -> bool;
    pub fn cogent_llama_state_seq_get_data_ext_alloc(
        context: *mut llama_context,
        seq_id: llama_seq_id,
        flags: u32,
        data_out: *mut *mut u8,
        size_out: *mut usize,
    ) -> bool;
    pub fn cogent_llama_state_seq_set_data_ext(
        context: *mut llama_context,
        seq_id: llama_seq_id,
        flags: u32,
        data: *const u8,
        size: usize,
    ) -> bool;
    pub fn cogent_common_checkpoint_capture(
        context: *mut llama_context,
        seq_id: llama_seq_id,
        flags: u32,
        error_out: *mut *mut c_char,
    ) -> *mut cogent_common_checkpoint;
    pub fn cogent_common_checkpoint_restore(
        checkpoint: *const cogent_common_checkpoint,
        context: *mut llama_context,
        seq_id: llama_seq_id,
        flags: u32,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn cogent_common_checkpoint_size(checkpoint: *const cogent_common_checkpoint) -> usize;
    pub fn cogent_common_checkpoint_free(checkpoint: *mut cogent_common_checkpoint);
    pub fn cogent_set_llama_log_quiet(quiet: bool);
    pub fn cogent_backend_load_all();
    pub fn cogent_backend_observability_json(include_details: bool) -> *mut c_char;
    pub fn cogent_llama_set_sampler(
        context: *mut llama_context,
        seq_id: llama_seq_id,
        sampler: *mut llama_sampler,
    ) -> bool;
    pub fn cogent_llama_decode(context: *mut llama_context, batch: *const llama_batch) -> c_int;
    pub fn cogent_llama_encode(context: *mut llama_context, batch: *const llama_batch) -> c_int;
    pub fn cogent_llama_embeddings_seq(
        context: *mut llama_context,
        seq_id: llama_seq_id,
    ) -> *const c_float;
    pub fn cogent_llama_embeddings_ith(
        context: *mut llama_context,
        i: c_int,
    ) -> *const c_float;
    pub fn cogent_llama_synchronize(context: *mut llama_context) -> bool;
    pub fn cogent_llama_sampler_sample(
        sampler: *mut llama_sampler,
        context: *mut llama_context,
        idx: c_int,
    ) -> llama_token;
    pub fn cogent_llama_sampler_accept(sampler: *mut llama_sampler, token: llama_token) -> bool;
    pub fn cogent_mtmd_default_marker() -> *const c_char;
    pub fn cogent_mtmd_init_from_file(
        mmproj_path: *const c_char,
        text_model: *const llama_model,
        use_gpu: bool,
        n_threads: c_int,
    ) -> *mut cogent_mtmd_context;
    pub fn cogent_mtmd_free(context: *mut cogent_mtmd_context);
    pub fn cogent_mtmd_support_vision(context: *const cogent_mtmd_context) -> bool;
    pub fn cogent_mtmd_bitmap_init_from_buf(
        context: *mut cogent_mtmd_context,
        data: *const u8,
        len: usize,
    ) -> *mut cogent_mtmd_bitmap;
    pub fn cogent_mtmd_bitmap_free(bitmap: *mut cogent_mtmd_bitmap);
    pub fn cogent_mtmd_input_chunks_init() -> *mut cogent_mtmd_input_chunks;
    pub fn cogent_mtmd_input_chunks_free(chunks: *mut cogent_mtmd_input_chunks);
    pub fn cogent_mtmd_tokenize(
        context: *mut cogent_mtmd_context,
        chunks: *mut cogent_mtmd_input_chunks,
        text: *const c_char,
        add_special: bool,
        parse_special: bool,
        bitmaps: *const *const cogent_mtmd_bitmap,
        bitmap_count: usize,
    ) -> bool;
    pub fn cogent_mtmd_eval_chunks(
        context: *mut cogent_mtmd_context,
        llama_context: *mut llama_context,
        chunks: *const cogent_mtmd_input_chunks,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
        new_n_past: *mut i32,
    ) -> i32;
}
