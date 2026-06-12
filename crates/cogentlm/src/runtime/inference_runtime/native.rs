//! Native-runtime convenience methods on `InferenceRuntime`.

use crate::error::Result;
use crate::native_bridge;

use super::InferenceRuntime;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/native_tests.rs"]
mod native_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

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

    pub fn media_marker(&self) -> Result<String> {
        Ok(native_bridge::mtmd_default_marker())
    }
}
