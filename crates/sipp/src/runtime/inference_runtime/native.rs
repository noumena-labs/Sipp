//! Native-runtime convenience methods on `InferenceRuntime`.

use crate::error::{Error, Result};
use crate::native_bridge::{self, NativeAudio};
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus, ResponseOutput};
use crate::runtime::{RequestStepResult, SynthesizedAudio};

use super::{ActiveSpeechRequest, InferenceRuntime};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/native_tests.rs"]
mod native_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const SPEECH_TEXT_REQUIRED: &str = "speech text must not be empty";
const SPEECH_DURATION_POSITIVE: &str = "max_duration_ms must be positive";
const SPEECH_REQUIRES_IDLE_RUNTIME: &str = "speech synthesis requires an idle runtime";
const TEXT_REQUIRES_IDLE_SPEECH: &str = "text inference cannot start during speech synthesis";

impl InferenceRuntime {
    pub fn get_bos_text(&self) -> Result<String> {
        let bos = self.native_runtime.bos_token();
        if bos == native_bridge::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        self.native_runtime.token_to_piece(bos, true)
    }

    pub fn get_eos_text(&self) -> Result<String> {
        let eos = self.native_runtime.eos_token();
        if eos == native_bridge::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        self.native_runtime.token_to_piece(eos, true)
    }

    pub fn chat_template_source(&self) -> Result<Option<String>> {
        self.native_runtime.chat_template_source().map(Some)
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        add_assistant: bool,
    ) -> Result<String> {
        self.native_runtime
            .apply_chat_template_json(messages_json, add_assistant)
    }

    pub(crate) fn apply_asr_chat_template(&self, language: &str) -> Result<String> {
        self.native_runtime.apply_asr_chat_template(language)
    }

    /// Build the model-specific prompt used for speech recognition.
    pub fn prepare_asr_prompt(&self, language: &str) -> Result<String> {
        self.apply_asr_chat_template(language)
    }

    pub(crate) fn parse_asr_output(&self, language: &str, output: &str) -> Result<String> {
        self.native_runtime.parse_asr_output(language, output)
    }

    /// Parse a model-specific speech-recognition response into transcript text.
    pub fn parse_asr_transcript(&self, language: &str, output: &str) -> Result<String> {
        self.parse_asr_output(language, output)
    }

    /// Enqueues speech synthesis into the shared inference scheduler.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime is not ready, the text is empty, the
    /// runtime already has an active request, or native setup fails.
    pub fn enqueue_speech_request(
        &mut self,
        text: &str,
        language: Option<&str>,
        speaker_audio: Option<&[u8]>,
        max_duration_ms: Option<u32>,
    ) -> Result<u32> {
        if !self.is_ready() {
            return Err(Error::RuntimeNotReady);
        }
        if text.is_empty() {
            return Err(Error::InvalidRequest(SPEECH_TEXT_REQUIRED));
        }
        if matches!(max_duration_ms, Some(0)) {
            return Err(Error::InvalidRequest(SPEECH_DURATION_POSITIVE));
        }
        if self.active_speech.is_some() || self.request_queue.has_uncompleted_requests() {
            return Err(Error::InvalidRequest(SPEECH_REQUIRES_IDLE_RUNTIME));
        }

        let request_id = self.next_generate_request_id()?;
        self.native_runtime
            .begin_speech(text, language, speaker_audio, max_duration_ms)?;
        self.request_queue.admit_external_request(request_id);
        self.active_speech = Some(ActiveSpeechRequest { request_id });
        Ok(request_id)
    }

    pub(super) fn ensure_speech_idle(&self) -> Result<()> {
        if self.active_speech.is_some() {
            return Err(Error::InvalidRequest(TEXT_REQUIRES_IDLE_SPEECH));
        }
        Ok(())
    }

    pub(super) fn run_speech_scheduler_tick_locked(&mut self) -> RequestStepResult {
        let Some(active) = self.active_speech else {
            return RequestStepResult::Waiting;
        };

        if self
            .request_queue
            .request_cancel_requested(active.request_id)
        {
            let response = match self.native_runtime.cancel_speech() {
                Ok(()) => GenerateResponse::cancelled(
                    active.request_id,
                    crate::runtime::REQUEST_CANCELLED_MESSAGE,
                ),
                Err(error) => GenerateResponse::failed(active.request_id, error.to_string()),
            };
            self.complete_speech_request(response);
            return RequestStepResult::Terminal;
        }

        match self.native_runtime.step_speech() {
            Ok(false) => RequestStepResult::Progressed,
            Ok(true) => {
                let response = match self.native_runtime.finish_speech() {
                    Ok(audio) => GenerateResponse::terminal(
                        active.request_id,
                        GenerateResponseStatus::Completed,
                        ResponseOutput::Audio(synthesized_audio(audio)),
                        String::new(),
                    ),
                    Err(error) => GenerateResponse::failed(active.request_id, error.to_string()),
                };
                self.complete_speech_request(response);
                RequestStepResult::Terminal
            }
            Err(error) => {
                self.complete_speech_request(GenerateResponse::failed(
                    active.request_id,
                    error.to_string(),
                ));
                RequestStepResult::Terminal
            }
        }
    }

    fn complete_speech_request(&mut self, response: GenerateResponse) {
        self.active_speech = None;
        self.request_queue.mark_completed(response);
    }

    /// Synthesizes speech through the loaded native audio-generation model.
    ///
    /// # Errors
    ///
    /// Returns an error when the request cannot be enqueued, native generation
    /// fails, or the scheduler stops before producing a terminal audio result.
    pub fn synthesize_speech(
        &mut self,
        text: &str,
        language: Option<&str>,
        speaker_audio: Option<&[u8]>,
        max_duration_ms: Option<u32>,
    ) -> Result<SynthesizedAudio> {
        let request_id =
            self.enqueue_speech_request(text, language, speaker_audio, max_duration_ms)?;
        while !self
            .request_queue
            .completed_responses
            .contains_key(&request_id)
        {
            match self.run_scheduler_tick() {
                RequestStepResult::Invalid => {
                    return Err(Error::RuntimeCommand(
                        "speech scheduler became invalid".to_string(),
                    ));
                }
                RequestStepResult::FatalNoProgress => {
                    return Err(Error::RuntimeCommand(
                        "speech scheduler failed to make progress".to_string(),
                    ));
                }
                RequestStepResult::Waiting => {
                    return Err(Error::RuntimeCommand(
                        "speech scheduler stopped before completion".to_string(),
                    ));
                }
                RequestStepResult::Progressed | RequestStepResult::Terminal => {}
            }
        }
        let response = self
            .take_completed_response(request_id)
            .ok_or_else(|| Error::RuntimeCommand("speech response is missing".to_string()))?;
        match (response.status, response.output) {
            (GenerateResponseStatus::Completed, ResponseOutput::Audio(audio)) => Ok(audio),
            (_, _) => Err(Error::RuntimeCommand(response.error_message)),
        }
    }

    pub fn media_marker(&self) -> Result<String> {
        Ok(native_bridge::mtmd_default_marker())
    }
}

fn synthesized_audio(audio: NativeAudio) -> SynthesizedAudio {
    SynthesizedAudio {
        data: audio.data,
        sample_count: audio.sample_count,
        sample_rate_hz: audio.sample_rate_hz,
    }
}
