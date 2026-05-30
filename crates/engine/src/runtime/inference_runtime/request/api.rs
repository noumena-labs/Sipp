use crate::collection::sorted_unique_non_empty_strings;
use crate::engine::protocol::EmbedOptions;
use crate::error::{Error, Result};
use crate::runtime::config::RequestSampling;
use crate::runtime::llama_token;
use crate::runtime::request::{
    GenerateRequest, GenerateRequestId, GenerateTokenEmissionMode, MultimodalPayload,
};
use crate::token::tokenize;

use super::super::{clamp_usize_to_i32, InferenceRuntime, DEFAULT_PROMPT_CONTEXT_KEY};

const N_TOKENS_PREDICT_POSITIVE: &str = "n_tokens_predict must be positive";
const IMAGE_BUFFERS_REQUIRED: &str = "image_buffers must not be empty";
const REQUEST_ID_OVERFLOW: &str = "request id overflow";
const FAILED_TO_ENQUEUE_REQUEST: &str = "failed to enqueue request";

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
        sampling: Option<RequestSampling>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest(N_TOKENS_PREDICT_POSITIVE));
        }
        self.text_generation_slot_plan()?;

        let request = self.prepare_generate_request(GenerateRequestInput {
            context_key: context_key.into(),
            prompt: prompt.into(),
            n_tokens_predict,
            grammar: grammar.into(),
            json_schema: json_schema.into(),
            stop,
            sampling,
            token_emission_mode,
            tokenization: RequestTokenization::Text,
        });
        self.enqueue_prepared_request(request?)
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
        sampling: Option<RequestSampling>,
        token_emission_mode: GenerateTokenEmissionMode,
    ) -> Result<GenerateRequestId> {
        if !self.is_ready() || self.mtmd_context.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        if n_tokens_predict <= 0 {
            return Err(Error::InvalidRequest(N_TOKENS_PREDICT_POSITIVE));
        }
        if image_buffers.is_empty() {
            return Err(Error::InvalidRequest(IMAGE_BUFFERS_REQUIRED));
        }
        self.text_generation_slot_plan()?;

        let mut request = self.prepare_generate_request(GenerateRequestInput {
            context_key: context_key.into(),
            prompt: prompt.into(),
            n_tokens_predict,
            grammar: grammar.into(),
            json_schema: json_schema.into(),
            stop,
            sampling,
            token_emission_mode,
            tokenization: RequestTokenization::Multimodal,
        })?;
        request.multimodal = Some(MultimodalPayload { image_buffers });
        request.is_multimodal_turn = true;
        self.enqueue_prepared_request(request)
    }

    pub fn enqueue_embed_request(
        &mut self,
        input: impl Into<String>,
        embed_options: EmbedOptions,
    ) -> Result<GenerateRequestId> {
        // Capability check first so the caller sees the most informative
        // error (UnsupportedOperation with a model-class reason) rather than
        // a generic "runtime not ready" when both apply.
        self.embedding_slot_plan()?;
        if !self.is_ready() {
            return Err(Error::RuntimeNotReady);
        }

        let context_key =
            normalize_context_key(embed_options.context_key.clone().unwrap_or_default());
        let input: String = input.into();

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &input, true, true)?;
        if prompt_tokens.is_empty() {
            return Err(Error::Tokenize);
        }

        let request_id = self.next_generate_request_id()?;
        let mut request = GenerateRequest::new(request_id, context_key);
        request.original_prompt = input;
        request.prompt_tokens = prompt_tokens;
        request.max_output_tokens = 0;
        request.embed_options = Some(embed_options);
        self.enqueue_prepared_request(request)
    }

    fn prepare_generate_request(&mut self, input: GenerateRequestInput) -> Result<GenerateRequest> {
        let context_key = normalize_context_key(input.context_key);
        let prompt = input.prompt;
        let grammar = input.grammar;
        let json_schema = input.json_schema;

        let vocab = self.vocab()?;
        let prompt_tokens = tokenize(vocab, &prompt, input.tokenization.add_bos(), true)?;
        if input.tokenization.requires_prompt_tokens() && prompt_tokens.is_empty() {
            return Err(Error::Tokenize);
        }

        let request_id = self.next_generate_request_id()?;

        Ok(generate_request(GenerateRequestFields {
            request_id,
            context_key,
            prompt,
            prompt_tokens,
            n_tokens_predict: input.n_tokens_predict,
            grammar,
            json_schema,
            stop: input.stop,
            sampling: input.sampling,
            token_emission_mode: input.token_emission_mode,
        }))
    }

    fn next_generate_request_id(&mut self) -> Result<GenerateRequestId> {
        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or(Error::InvalidRequest(REQUEST_ID_OVERFLOW))?;
        Ok(request_id)
    }

    fn enqueue_prepared_request(
        &mut self,
        mut request: GenerateRequest,
    ) -> Result<GenerateRequestId> {
        let request_id = request.id;
        request.input_tokens = clamp_usize_to_i32(request.prompt_tokens.len());
        self.total_input_tokens = self
            .total_input_tokens
            .saturating_add(request.prompt_tokens.len());

        if !self.request_queue.push(request) {
            return Err(Error::InvalidRequest(FAILED_TO_ENQUEUE_REQUEST));
        }

        Ok(request_id)
    }
}

struct GenerateRequestInput {
    context_key: String,
    prompt: String,
    n_tokens_predict: i32,
    grammar: String,
    json_schema: String,
    stop: Vec<String>,
    sampling: Option<RequestSampling>,
    token_emission_mode: GenerateTokenEmissionMode,
    tokenization: RequestTokenization,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RequestTokenization {
    Text,
    Multimodal,
}

impl RequestTokenization {
    fn add_bos(self) -> bool {
        matches!(self, Self::Text)
    }

    fn requires_prompt_tokens(self) -> bool {
        matches!(self, Self::Text)
    }
}

struct GenerateRequestFields {
    request_id: GenerateRequestId,
    context_key: String,
    prompt: String,
    prompt_tokens: Vec<llama_token>,
    n_tokens_predict: i32,
    grammar: String,
    json_schema: String,
    stop: Vec<String>,
    sampling: Option<RequestSampling>,
    token_emission_mode: GenerateTokenEmissionMode,
}

fn generate_request(fields: GenerateRequestFields) -> GenerateRequest {
    let mut request = GenerateRequest::new(fields.request_id, fields.context_key);
    request.original_prompt = fields.prompt;
    request.prompt_tokens = fields.prompt_tokens;
    request.max_output_tokens = fields.n_tokens_predict;
    request.token_emission_mode = fields.token_emission_mode;
    request.grammar = fields.grammar;
    request.json_schema = fields.json_schema;
    request.stop = normalize_stop_sequences(fields.stop);
    request.sampling = fields.sampling;
    request
}

fn normalize_context_key(context_key: impl Into<String>) -> String {
    let context_key = context_key.into();
    if context_key.is_empty() {
        DEFAULT_PROMPT_CONTEXT_KEY.to_string()
    } else {
        context_key
    }
}

pub(super) fn normalize_stop_sequences(stop: Vec<String>) -> Vec<String> {
    sorted_unique_non_empty_strings(stop)
}

#[cfg(test)]
pub(super) fn request_tokenization_flags_for_tests(tokenization: &str) -> Option<(bool, bool)> {
    let tokenization = match tokenization {
        "text" => RequestTokenization::Text,
        "multimodal" => RequestTokenization::Multimodal,
        _ => return None,
    };
    Some((
        tokenization.add_bos(),
        tokenization.requires_prompt_tokens(),
    ))
}
