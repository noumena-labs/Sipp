//! Tests the `native_bridge` module in `cogentlm-engine`.
//!
//! Covers null native handles and empty sampler fallbacks deterministically,
//! without loading a model or calling backend-global FFI entry points.

use super::*;

#[test]
fn empty_runtime_handle_reports_safe_defaults() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();

    assert!(!runtime.is_loaded());
    assert_eq!(runtime.n_ctx(), 0);
    assert_eq!(runtime.n_batch(), 0);
    assert_eq!(runtime.n_ubatch(), 0);
    assert_eq!(runtime.n_seq_max(), 0);
    assert_eq!(runtime.n_embd_out(), 0);
    assert_eq!(runtime.n_cls_out(), 0);
    assert_eq!(runtime.pooling_type(), 0);
    assert!(!runtime.has_encoder());
    assert!(!runtime.has_decoder());
    assert!(!runtime.has_chat_template());
    assert!(!runtime.is_recurrent());
    assert!(!runtime.is_hybrid());
    assert!(!runtime.kv_unified());
    assert_eq!(runtime.flash_attention(), "unknown");
    assert_eq!(runtime.cache_type_k(), "unknown");
    assert_eq!(runtime.cache_type_v(), "unknown");
    assert_eq!(runtime.bos_token(), LLAMA_TOKEN_NULL);
    assert_eq!(runtime.eos_token(), LLAMA_TOKEN_NULL);
    assert_eq!(runtime.decoder_start_token(), LLAMA_TOKEN_NULL);
    assert!(!runtime.is_eog(0));
    assert!(!runtime.mtmd_ready());
    assert!(!runtime.mtmd_support_vision());

    let limits = runtime.resolved_limits();
    assert_eq!(limits.n_ctx, 0);
    assert_eq!(limits.flash_attention, "unknown");

    assert!(!runtime.synchronize());
    assert!(!runtime.clear_sequence(0, 0, -1));
    runtime.add_sequence_delta(0, 0, -1, -1);
    assert!(!runtime.set_state_seq(0, &[1, 2, 3]));
    assert!(!runtime.init_mtmd("projector.gguf", false, 1));
    assert!(!runtime.detach_sampler(0));
}

#[test]
fn empty_runtime_handle_rejects_required_native_calls() {
    let runtime = NativeRuntimeHandle::empty_for_tests();

    assert!(matches!(
        runtime.chat_template_source(),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.tokenize("hello", true, false),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.token_to_piece(1, true),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.token_to_piece_bytes(1, true),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.apply_chat_template_json("[]", true),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.embeddings_seq(0),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(runtime.state_seq(0), Err(Error::RuntimeNotReady)));
    assert!(matches!(
        runtime.create_sampler("{}", "", ""),
        Err(Error::RuntimeNotReady)
    ));
}

#[test]
fn empty_sampler_handle_is_inert() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut sampler = SamplerHandle::empty_for_tests();

    assert!(!sampler.backend_sampling());
    assert!(!sampler.accept(1, true));
    sampler.reset();
    assert_eq!(runtime.sample_with(&mut sampler, 0), LLAMA_TOKEN_NULL);
    assert!(!runtime.attach_sampler(&mut sampler, 0));
    assert!(!format!("{sampler:?}").is_empty());
}
