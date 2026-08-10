#[allow(
    clippy::too_many_arguments,
    reason = "CXX bridge signatures mirror the native ABI"
)]
#[cxx::bridge(namespace = "sipp::sys")]
pub mod ffi {
    extern "Rust" {
        fn extend_bytes(output: &mut Vec<u8>, data: &[u8]);
        fn extend_floats(output: &mut Vec<f32>, data: &[f32]);
    }

    unsafe extern "C++" {
        include!("sipp_cxx.h");

        type NativeRuntime;
        type NativeBatch;
        type NativeAudio;
        type CommonSampler;

        fn backend_init();
        fn backend_load_all();
        fn set_llama_log_quiet(quiet: bool);
        fn backend_observability_json(include_details: bool) -> String;
        fn mtmd_default_marker() -> String;

        fn load_native_runtime(
            model_path: &str,
            args_json: &str,
        ) -> Result<UniquePtr<NativeRuntime>>;
        fn n_ctx(self: &NativeRuntime) -> i32;
        fn n_batch(self: &NativeRuntime) -> i32;
        fn n_ubatch(self: &NativeRuntime) -> i32;
        fn n_seq_max(self: &NativeRuntime) -> i32;
        fn n_threads(self: &NativeRuntime) -> i32;
        fn n_threads_batch(self: &NativeRuntime) -> i32;
        fn n_embd_out(self: &NativeRuntime) -> i32;
        fn n_cls_out(self: &NativeRuntime) -> i32;
        fn pooling_type(self: &NativeRuntime) -> i32;
        fn has_encoder(self: &NativeRuntime) -> bool;
        fn has_decoder(self: &NativeRuntime) -> bool;
        fn has_chat_template(self: &NativeRuntime) -> bool;
        fn is_recurrent(self: &NativeRuntime) -> bool;
        fn is_hybrid(self: &NativeRuntime) -> bool;
        fn kv_unified(self: &NativeRuntime) -> bool;
        fn flash_attention(self: &NativeRuntime) -> String;
        fn cache_type_k(self: &NativeRuntime) -> String;
        fn cache_type_v(self: &NativeRuntime) -> String;
        fn chat_template_source(self: &NativeRuntime) -> String;
        fn bos_token(self: &NativeRuntime) -> i32;
        fn eos_token(self: &NativeRuntime) -> i32;
        fn decoder_start_token(self: &NativeRuntime) -> i32;
        fn is_eog(self: &NativeRuntime, token: i32) -> bool;
        fn mtmd_ready(self: &NativeRuntime) -> bool;
        fn tokenize(
            self: &NativeRuntime,
            text: &str,
            add_special: bool,
            parse_special: bool,
        ) -> Result<Vec<i32>>;
        fn token_to_piece(self: &NativeRuntime, token: i32, special: bool) -> Result<String>;
        fn token_to_piece_bytes_into(
            self: &NativeRuntime,
            token: i32,
            special: bool,
            out: &mut Vec<u8>,
        ) -> Result<()>;
        fn apply_chat_template_json(
            self: &NativeRuntime,
            messages_json: &str,
            add_assistant: bool,
        ) -> Result<String>;
        fn apply_asr_chat_template(self: &NativeRuntime, language: &str) -> Result<String>;
        fn parse_asr_output(self: &NativeRuntime, language: &str, output: &str) -> Result<String>;
        fn begin_speech(
            self: Pin<&mut NativeRuntime>,
            text: &str,
            language: &str,
            voice: &[u8],
            has_max_duration: bool,
            max_duration_ms: u32,
        ) -> Result<()>;
        fn step_speech(self: Pin<&mut NativeRuntime>) -> Result<bool>;
        fn finish_speech(self: Pin<&mut NativeRuntime>) -> Result<UniquePtr<NativeAudio>>;
        fn cancel_speech(self: Pin<&mut NativeRuntime>) -> Result<()>;
        fn take_data(self: Pin<&mut NativeAudio>) -> Vec<u8>;
        fn sample_count(self: &NativeAudio) -> u64;
        fn sample_rate_hz(self: &NativeAudio) -> u32;
        fn decode(self: Pin<&mut NativeRuntime>, batch: &NativeBatch) -> Result<i32>;
        fn encode(self: Pin<&mut NativeRuntime>, batch: &NativeBatch) -> Result<i32>;
        fn synchronize(self: Pin<&mut NativeRuntime>) -> bool;
        fn clear_sequence(self: Pin<&mut NativeRuntime>, seq_id: i32, p0: i32, p1: i32) -> bool;
        fn add_sequence_delta(
            self: Pin<&mut NativeRuntime>,
            seq_id: i32,
            p0: i32,
            p1: i32,
            delta: i32,
        );
        fn embeddings_seq(self: &NativeRuntime, seq_id: i32) -> Result<Vec<f32>>;
        fn state_seq(self: &NativeRuntime, seq_id: i32) -> Result<Vec<u8>>;
        fn set_state_seq(self: Pin<&mut NativeRuntime>, seq_id: i32, data: &[u8]) -> bool;
        fn init_mtmd(
            self: Pin<&mut NativeRuntime>,
            projector_path: &str,
            use_gpu: bool,
            n_threads: i32,
        ) -> bool;
        fn mtmd_support_vision(self: &NativeRuntime) -> bool;
        fn mtmd_audio_sample_rate(self: &NativeRuntime) -> i32;
        fn mtmd_generated_audio_sample_rate(self: &NativeRuntime) -> i32;
        fn mtmd_eval_media(
            self: Pin<&mut NativeRuntime>,
            prompt: &str,
            media_bytes: &[u8],
            media_sizes: &[i32],
            add_special: bool,
            parse_special: bool,
            n_past: i32,
            seq_id: i32,
            n_batch: i32,
            logits_last: bool,
        ) -> Result<i32>;

        fn make_native_batch() -> UniquePtr<NativeBatch>;
        fn ensure_capacity(
            self: Pin<&mut NativeBatch>,
            max_tokens: i32,
            max_sequences: i32,
        ) -> Result<()>;
        fn reset(self: Pin<&mut NativeBatch>);
        fn add_token(
            self: Pin<&mut NativeBatch>,
            token: i32,
            pos: i32,
            seq_id: i32,
            logits: bool,
        ) -> bool;
        fn n_tokens(self: &NativeBatch) -> i32;
        fn token(self: &NativeBatch, index: i32) -> i32;
        fn pos(self: &NativeBatch, index: i32) -> i32;
        fn seq_id(self: &NativeBatch, index: i32) -> i32;
        fn logits(self: &NativeBatch, index: i32) -> bool;
        fn clear_logits(self: Pin<&mut NativeBatch>);
        fn set_last_logits(self: Pin<&mut NativeBatch>);

        fn create_sampler(
            runtime: &NativeRuntime,
            sampling_json: &str,
            grammar: &str,
            json_schema: &str,
        ) -> Result<UniquePtr<CommonSampler>>;
        fn sampler_accept(self: Pin<&mut CommonSampler>, token: i32, accept_grammar: bool) -> bool;
        fn sampler_reset(self: Pin<&mut CommonSampler>);
        fn sampler_backend_sampling(self: &CommonSampler) -> bool;
        fn sampler_sample(
            sampler: Pin<&mut CommonSampler>,
            runtime: Pin<&mut NativeRuntime>,
            idx: i32,
        ) -> i32;
        fn sampler_attach(
            sampler: Pin<&mut CommonSampler>,
            runtime: Pin<&mut NativeRuntime>,
            seq_id: i32,
        ) -> bool;
        fn sampler_detach(runtime: Pin<&mut NativeRuntime>, seq_id: i32) -> bool;
    }
}

fn extend_bytes(output: &mut Vec<u8>, data: &[u8]) {
    output.extend_from_slice(data);
}

fn extend_floats(output: &mut Vec<f32>, data: &[f32]) {
    output.extend_from_slice(data);
}

pub use ffi::*;
