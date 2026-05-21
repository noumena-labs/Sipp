use crate::error::{Error, Result};
use crate::runtime::config::SamplingRuntimeConfig;
use crate::runtime::request::{GenerateRequest, GenerateRequestId, GenerateTokenEmissionMode};
use crate::token::tokenize;

use super::super::{clamp_usize_to_i32, InferenceRuntime, DEFAULT_PROMPT_CONTEXT_KEY};

impl InferenceRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_request(
        &mut self,
        context_key: impl Into<String>,
        prompt: impl Into<String>,
        n_tokens_predict: i32,
        grammar: impl Into<String>,
        json_schema: impl Into<String>,
        stop: Vec<String>,
        sampling: Option<SamplingRuntimeConfig>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest("n_tokens_predict must be positive"));
        }

        let mut context_key = context_key.into();
        if context_key.is_empty() {
            context_key = DEFAULT_PROMPT_CONTEXT_KEY.to_string();
        }
        let prompt = prompt.into();
        let grammar = grammar.into();
        let json_schema = json_schema.into();

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &prompt, true, true)?;
        if prompt_tokens.is_empty() {
            return Err(Error::Tokenize);
        }

        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(Error::InvalidRequest("request id overflow"))?;

        let mut request = GenerateRequest::new(request_id, context_key);
        request.original_prompt = prompt;
        request.max_output_tokens = n_tokens_predict;
        request.token_emission_mode = token_emission_mode;
        request.prompt_tokens = prompt_tokens;
        request.grammar = grammar;
        request.json_schema = json_schema;
        request.stop = normalize_stop_sequences(stop);
        request.sampling = sampling;
        request.input_tokens = clamp_usize_to_i32(request.prompt_tokens.len());
        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(request.prompt_tokens.len());

        if !self.request_queue.push(request) {
            return Err(Error::InvalidRequest("failed to enqueue request"));
        }

        Ok(request_id)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn enqueue_multimodal_request(
        &mut self,
        context_key: impl Into<String>,
        prompt: impl Into<String>,
        n_tokens_predict: i32,
        image_buffers: Vec<Vec<u8>>,
        grammar: impl Into<String>,
        json_schema: impl Into<String>,
        stop: Vec<String>,
        sampling: Option<SamplingRuntimeConfig>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() || self.mtmd_context.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest("n_tokens_predict must be positive"));
        }
        if image_buffers.is_empty() {
            return Err(Error::InvalidRequest("image_buffers must not be empty"));
        }

        let mut context_key = context_key.into();
        if context_key.is_empty() {
            context_key = DEFAULT_PROMPT_CONTEXT_KEY.to_string();
        }
        let prompt = prompt.into();
        let grammar = grammar.into();
        let json_schema = json_schema.into();

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &prompt, false, true)?;

        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(Error::InvalidRequest("request id overflow"))?;

        let mut request = GenerateRequest::new(request_id, context_key);
        request.original_prompt = prompt;
        request.prompt_tokens = prompt_tokens;
        request.multimodal = Some(crate::runtime::request::MultimodalPayload { image_buffers });
        request.max_output_tokens = n_tokens_predict;
        request.token_emission_mode = token_emission_mode;
        request.is_multimodal_turn = true;
        request.grammar = grammar;
        request.json_schema = json_schema;
        request.stop = normalize_stop_sequences(stop);
        request.sampling = sampling;
        request.input_tokens = clamp_usize_to_i32(request.prompt_tokens.len());
        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(request.prompt_tokens.len());

        if !self.request_queue.push(request) {
            return Err(Error::InvalidRequest("failed to enqueue request"));
        }

        Ok(request_id)
    }
}

pub(super) fn normalize_stop_sequences(stop: Vec<String>) -> Vec<String> {
    let mut normalized = Vec::with_capacity(stop.len());
    normalized.extend(stop.into_iter().filter(|value| !value.is_empty()));
    normalized.sort();
    normalized.dedup();
    normalized
}
