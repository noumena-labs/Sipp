//! Crate-private facade over the CXX bridge.
//!
//! Engine modules should depend on these handles and helpers instead of
//! naming `cogentlm_sys::bridge` types directly.

use std::pin::Pin;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};
use crate::runtime::config::ResolvedRuntimeLimits;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/native_bridge_tests.rs"]
mod native_bridge_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) const LLAMA_TOKEN_NULL: ffi::llama_token = ffi::LLAMA_TOKEN_NULL;

pub(crate) fn backend_init() {
    ffi::bridge::backend_init();
}

pub(crate) fn backend_load_all() {
    ffi::bridge::backend_load_all();
}

pub(crate) fn set_llama_log_quiet(quiet: bool) {
    ffi::bridge::set_llama_log_quiet(quiet);
}

pub(crate) fn backend_observability_json(include_details: bool) -> String {
    ffi::bridge::backend_observability_json(include_details)
}

pub(crate) fn mtmd_default_marker() -> String {
    ffi::bridge::mtmd_default_marker()
}

pub(crate) struct NativeRuntimeHandle {
    inner: cxx::UniquePtr<ffi::bridge::NativeRuntime>,
}

impl NativeRuntimeHandle {
    pub(crate) fn load(model_path: &str, args_json: &str) -> Result<Self> {
        let inner = ffi::bridge::load_native_runtime(model_path, args_json)
            .map_err(|error| Error::RuntimeCommand(error.to_string()))?;
        if inner.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        Ok(Self { inner })
    }

    #[cfg(test)]
    pub(crate) fn empty_for_tests() -> Self {
        Self {
            inner: cxx::UniquePtr::null(),
        }
    }

    pub(crate) fn is_loaded(&self) -> bool {
        !self.inner.is_null()
    }

    pub(crate) fn resolved_limits(&self) -> ResolvedRuntimeLimits {
        ResolvedRuntimeLimits {
            n_ctx: self.n_ctx().max(0),
            n_batch: self.n_batch().max(0),
            n_ubatch: self.n_ubatch().max(0),
            n_parallel: self.n_seq_max().max(0),
            kv_unified: self.kv_unified(),
            flash_attention: self.flash_attention(),
            cache_type_k: self.cache_type_k(),
            cache_type_v: self.cache_type_v(),
        }
    }

    pub(crate) fn n_ctx(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_ctx()).unwrap_or(0)
    }

    pub(crate) fn n_batch(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_batch()).unwrap_or(0)
    }

    pub(crate) fn n_ubatch(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_ubatch()).unwrap_or(0)
    }

    pub(crate) fn n_seq_max(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_seq_max()).unwrap_or(0)
    }

    pub(crate) fn n_embd_out(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_embd_out()).unwrap_or(0)
    }

    pub(crate) fn n_cls_out(&self) -> i32 {
        self.with_ref(|runtime| runtime.n_cls_out()).unwrap_or(0)
    }

    pub(crate) fn pooling_type(&self) -> i32 {
        self.with_ref(|runtime| runtime.pooling_type()).unwrap_or(0)
    }

    pub(crate) fn has_encoder(&self) -> bool {
        self.with_ref(|runtime| runtime.has_encoder())
            .unwrap_or(false)
    }

    pub(crate) fn has_decoder(&self) -> bool {
        self.with_ref(|runtime| runtime.has_decoder())
            .unwrap_or(false)
    }

    pub(crate) fn has_chat_template(&self) -> bool {
        self.with_ref(|runtime| runtime.has_chat_template())
            .unwrap_or(false)
    }

    pub(crate) fn is_recurrent(&self) -> bool {
        self.with_ref(|runtime| runtime.is_recurrent())
            .unwrap_or(false)
    }

    pub(crate) fn is_hybrid(&self) -> bool {
        self.with_ref(|runtime| runtime.is_hybrid())
            .unwrap_or(false)
    }

    pub(crate) fn kv_unified(&self) -> bool {
        self.with_ref(|runtime| runtime.kv_unified())
            .unwrap_or(false)
    }

    pub(crate) fn flash_attention(&self) -> String {
        self.with_ref(|runtime| runtime.flash_attention())
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub(crate) fn cache_type_k(&self) -> String {
        self.with_ref(|runtime| runtime.cache_type_k())
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub(crate) fn cache_type_v(&self) -> String {
        self.with_ref(|runtime| runtime.cache_type_v())
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub(crate) fn chat_template_source(&self) -> Result<String> {
        self.with_required_ref(|runtime| runtime.chat_template_source())
    }

    pub(crate) fn bos_token(&self) -> i32 {
        self.with_ref(|runtime| runtime.bos_token())
            .unwrap_or(LLAMA_TOKEN_NULL)
    }

    pub(crate) fn eos_token(&self) -> i32 {
        self.with_ref(|runtime| runtime.eos_token())
            .unwrap_or(LLAMA_TOKEN_NULL)
    }

    pub(crate) fn decoder_start_token(&self) -> i32 {
        self.with_ref(|runtime| runtime.decoder_start_token())
            .unwrap_or(LLAMA_TOKEN_NULL)
    }

    pub(crate) fn is_eog(&self, token: ffi::llama_token) -> bool {
        self.with_ref(|runtime| runtime.is_eog(token))
            .unwrap_or(false)
    }

    pub(crate) fn mtmd_ready(&self) -> bool {
        self.with_ref(|runtime| runtime.mtmd_ready())
            .unwrap_or(false)
    }

    pub(crate) fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<ffi::llama_token>> {
        self.with_required_ref(|runtime| {
            runtime
                .tokenize(text, add_special, parse_special)
                .map_err(|error| Error::RuntimeCommand(error.to_string()))
        })?
    }

    pub(crate) fn token_to_piece(&self, token: ffi::llama_token, special: bool) -> Result<String> {
        self.with_required_ref(|runtime| {
            runtime
                .token_to_piece(token, special)
                .map_err(|_| Error::TokenToPiece { token })
        })?
    }

    pub(crate) fn token_to_piece_bytes(
        &self,
        token: ffi::llama_token,
        special: bool,
    ) -> Result<Vec<u8>> {
        self.with_required_ref(|runtime| {
            runtime
                .token_to_piece_bytes(token, special)
                .map_err(|_| Error::TokenToPiece { token })
        })?
    }

    pub(crate) fn apply_chat_template_json(
        &self,
        messages_json: &str,
        add_assistant: bool,
    ) -> Result<String> {
        self.with_required_ref(|runtime| {
            runtime
                .apply_chat_template_json(messages_json, add_assistant)
                .map_err(|error| Error::RuntimeCommand(error.to_string()))
        })?
    }

    pub(crate) fn decode(&mut self, batch: &NativeBatchHandle) -> Result<i32> {
        let batch = batch.as_ref()?;
        self.pin_mut()?
            .decode(batch)
            .map_err(|error| Error::RuntimeCommand(error.to_string()))
    }

    pub(crate) fn encode(&mut self, batch: &NativeBatchHandle) -> Result<i32> {
        let batch = batch.as_ref()?;
        self.pin_mut()?
            .encode(batch)
            .map_err(|error| Error::RuntimeCommand(error.to_string()))
    }

    pub(crate) fn synchronize(&mut self) -> bool {
        let Ok(runtime) = self.pin_mut() else {
            return false;
        };
        runtime.synchronize()
    }

    pub(crate) fn clear_sequence(&mut self, seq_id: ffi::llama_seq_id, p0: i32, p1: i32) -> bool {
        let Ok(mut runtime) = self.pin_mut() else {
            return false;
        };
        runtime.as_mut().clear_sequence(seq_id, p0, p1)
    }

    pub(crate) fn add_sequence_delta(
        &mut self,
        seq_id: ffi::llama_seq_id,
        p0: i32,
        p1: i32,
        delta: i32,
    ) {
        if let Ok(mut runtime) = self.pin_mut() {
            runtime.as_mut().add_sequence_delta(seq_id, p0, p1, delta);
        }
    }

    pub(crate) fn embeddings_seq(&self, seq_id: ffi::llama_seq_id) -> Result<Vec<f32>> {
        self.with_required_ref(|runtime| {
            runtime
                .embeddings_seq(seq_id)
                .map_err(|error| Error::RuntimeCommand(error.to_string()))
        })?
    }

    pub(crate) fn state_seq(&self, seq_id: ffi::llama_seq_id) -> Result<Vec<u8>> {
        self.with_required_ref(|runtime| {
            runtime
                .state_seq(seq_id)
                .map_err(|error| Error::RuntimeCommand(error.to_string()))
        })?
    }

    pub(crate) fn set_state_seq(&mut self, seq_id: ffi::llama_seq_id, data: &[u8]) -> bool {
        let Ok(mut runtime) = self.pin_mut() else {
            return false;
        };
        runtime.as_mut().set_state_seq(seq_id, data)
    }

    pub(crate) fn init_mtmd(
        &mut self,
        projector_path: &str,
        use_gpu: bool,
        n_threads: i32,
    ) -> bool {
        let Ok(mut runtime) = self.pin_mut() else {
            return false;
        };
        runtime
            .as_mut()
            .init_mtmd(projector_path, use_gpu, n_threads)
    }

    pub(crate) fn mtmd_support_vision(&self) -> bool {
        self.with_ref(|runtime| runtime.mtmd_support_vision())
            .unwrap_or(false)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mtmd_eval_images(
        &mut self,
        prompt: &str,
        image_bytes: &[u8],
        image_sizes: &[i32],
        add_special: bool,
        parse_special: bool,
        n_past: i32,
        seq_id: ffi::llama_seq_id,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32> {
        self.pin_mut()?
            .mtmd_eval_images(
                prompt,
                image_bytes,
                image_sizes,
                add_special,
                parse_special,
                n_past,
                seq_id,
                n_batch,
                logits_last,
            )
            .map_err(|error| Error::RuntimeCommand(error.to_string()))
    }

    pub(crate) fn create_sampler(
        &self,
        sampling_json: &str,
        grammar: &str,
        json_schema: &str,
    ) -> Result<SamplerHandle> {
        self.with_required_ref(|runtime| {
            ffi::bridge::create_sampler(runtime, sampling_json, grammar, json_schema)
                .map(SamplerHandle::new)
                .map_err(|error| Error::RuntimeCommand(error.to_string()))
        })?
    }

    pub(crate) fn sample_with(
        &mut self,
        sampler: &mut SamplerHandle,
        idx: i32,
    ) -> ffi::llama_token {
        let Ok(runtime) = self.pin_mut() else {
            return LLAMA_TOKEN_NULL;
        };
        sampler.sample(runtime, idx)
    }

    pub(crate) fn attach_sampler(
        &mut self,
        sampler: &mut SamplerHandle,
        seq_id: ffi::llama_seq_id,
    ) -> bool {
        let Ok(runtime) = self.pin_mut() else {
            return false;
        };
        sampler.attach(runtime, seq_id)
    }

    pub(crate) fn detach_sampler(&mut self, seq_id: ffi::llama_seq_id) -> bool {
        let Ok(runtime) = self.pin_mut() else {
            return false;
        };
        ffi::bridge::sampler_detach(runtime, seq_id)
    }

    fn with_ref<T>(&self, f: impl FnOnce(&ffi::bridge::NativeRuntime) -> T) -> Option<T> {
        self.inner.as_ref().map(f)
    }

    fn with_required_ref<T>(&self, f: impl FnOnce(&ffi::bridge::NativeRuntime) -> T) -> Result<T> {
        self.inner.as_ref().map(f).ok_or(Error::RuntimeNotReady)
    }

    fn pin_mut(&mut self) -> Result<Pin<&mut ffi::bridge::NativeRuntime>> {
        if self.inner.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        Ok(self.inner.pin_mut())
    }
}

pub(crate) struct NativeBatchHandle {
    inner: cxx::UniquePtr<ffi::bridge::NativeBatch>,
}

impl NativeBatchHandle {
    pub(crate) fn new() -> Self {
        Self {
            inner: ffi::bridge::make_native_batch(),
        }
    }

    pub(crate) fn ensure_capacity(&mut self, max_tokens: i32, max_sequences: i32) -> Result<()> {
        if self.inner.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        self.inner
            .pin_mut()
            .ensure_capacity(max_tokens, max_sequences)
            .map_err(|_| Error::NullPointer("llama_batch_init"))
    }

    pub(crate) fn reset(&mut self) {
        if !self.inner.is_null() {
            self.inner.pin_mut().reset();
        }
    }

    pub(crate) fn add_token(
        &mut self,
        token: ffi::llama_token,
        position: i32,
        seq_id: ffi::llama_seq_id,
        request_logits: bool,
    ) -> bool {
        if self.inner.is_null() {
            return false;
        }
        self.inner
            .pin_mut()
            .add_token(token, position, seq_id, request_logits)
    }

    pub(crate) fn n_tokens(&self) -> i32 {
        self.inner
            .as_ref()
            .map(|batch| batch.n_tokens())
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn token(&self, index: i32) -> i32 {
        self.inner
            .as_ref()
            .map(|batch| batch.token(index))
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn pos(&self, index: i32) -> i32 {
        self.inner
            .as_ref()
            .map(|batch| batch.pos(index))
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn seq_id(&self, index: i32) -> i32 {
        self.inner
            .as_ref()
            .map(|batch| batch.seq_id(index))
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn logits(&self, index: i32) -> bool {
        self.inner
            .as_ref()
            .map(|batch| batch.logits(index))
            .unwrap_or(false)
    }

    fn as_ref(&self) -> Result<&ffi::bridge::NativeBatch> {
        self.inner.as_ref().ok_or(Error::RuntimeNotReady)
    }
}

pub(crate) struct SamplerHandle {
    inner: cxx::UniquePtr<ffi::bridge::CommonSampler>,
}

impl SamplerHandle {
    #[cfg(test)]
    pub(crate) fn empty_for_tests() -> Self {
        Self {
            inner: cxx::UniquePtr::null(),
        }
    }

    fn new(inner: cxx::UniquePtr<ffi::bridge::CommonSampler>) -> Self {
        Self { inner }
    }

    pub(crate) fn backend_sampling(&self) -> bool {
        self.inner
            .as_ref()
            .is_some_and(|sampler| sampler.sampler_backend_sampling())
    }

    pub(crate) fn accept(&mut self, token: ffi::llama_token, accept_grammar: bool) -> bool {
        if self.inner.is_null() {
            return false;
        }
        self.inner.pin_mut().sampler_accept(token, accept_grammar)
    }

    pub(crate) fn reset(&mut self) {
        if !self.inner.is_null() {
            self.inner.pin_mut().sampler_reset();
        }
    }

    fn sample(
        &mut self,
        runtime: Pin<&mut ffi::bridge::NativeRuntime>,
        idx: i32,
    ) -> ffi::llama_token {
        if self.inner.is_null() {
            return LLAMA_TOKEN_NULL;
        }
        ffi::bridge::sampler_sample(self.inner.pin_mut(), runtime, idx)
    }

    fn attach(
        &mut self,
        runtime: Pin<&mut ffi::bridge::NativeRuntime>,
        seq_id: ffi::llama_seq_id,
    ) -> bool {
        if self.inner.is_null() {
            return false;
        }
        ffi::bridge::sampler_attach(self.inner.pin_mut(), runtime, seq_id)
    }
}

impl std::fmt::Debug for SamplerHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SamplerHandle")
            .finish_non_exhaustive()
    }
}
