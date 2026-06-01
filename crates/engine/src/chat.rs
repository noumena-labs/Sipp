use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{Error, Result};
use crate::runtime::InferenceRuntime;

const BOUNDARY_SYSTEM_SENTINEL: &str = "__CE_BOUNDARY_SYSTEM__";
const BOUNDARY_USER1_SENTINEL: &str = "__CE_BOUNDARY_USER1__";
const BOUNDARY_ASSISTANT_SENTINEL: &str = "__CE_BOUNDARY_ASSISTANT__";
const BOUNDARY_USER2_SENTINEL: &str = "__CE_BOUNDARY_USER2__";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatBoundaryInfo {
    pub assistant_prefix: String,
    pub assistant_suffix: String,
    pub next_turn_prefixes: Vec<String>,
    pub eos_text: String,
}

pub fn default_media_marker() -> Result<String> {
    Ok(crate::native_bridge::mtmd_default_marker())
}

pub fn probe_chat_boundary_info(
    mut apply_chat_template_json: impl FnMut(&str, bool) -> Result<String>,
    eos_text: impl Into<String>,
) -> Result<ChatBoundaryInfo> {
    let system_message = template_message("system", BOUNDARY_SYSTEM_SENTINEL);
    let user1_message = template_message("user", BOUNDARY_USER1_SENTINEL);
    let assistant_message = template_message("assistant", BOUNDARY_ASSISTANT_SENTINEL);
    let user2_message = template_message("user", BOUNDARY_USER2_SENTINEL);

    let closed_user_prompt =
        apply_chat_template_json(&messages_json(&[&system_message, &user1_message])?, false)?;
    let primed_assistant_prompt =
        apply_chat_template_json(&messages_json(&[&system_message, &user1_message])?, true)?;
    let closed_assistant_prompt = apply_chat_template_json(
        &messages_json(&[&system_message, &user1_message, &assistant_message])?,
        false,
    )?;
    let prompt_with_next_user = apply_chat_template_json(
        &messages_json(&[
            &system_message,
            &user1_message,
            &assistant_message,
            &user2_message,
        ])?,
        false,
    )?;

    let assistant_prefix = primed_assistant_prompt
        .strip_prefix(&closed_user_prompt)
        .unwrap_or("")
        .to_string();
    let assistant_suffix =
        slice_after_sentinel(&closed_assistant_prompt, BOUNDARY_ASSISTANT_SENTINEL).to_string();
    let next_user_append = prompt_with_next_user
        .strip_prefix(&closed_assistant_prompt)
        .unwrap_or("");
    let next_user_prefix = slice_before_sentinel(next_user_append, BOUNDARY_USER2_SENTINEL);
    let system_prefix = slice_before_sentinel(&closed_user_prompt, BOUNDARY_SYSTEM_SENTINEL);

    Ok(ChatBoundaryInfo {
        assistant_prefix: assistant_prefix.clone(),
        assistant_suffix,
        next_turn_prefixes: unique_non_empty(&[system_prefix, next_user_prefix, &assistant_prefix]),
        eos_text: eos_text.into(),
    })
}

impl InferenceRuntime {
    pub fn probe_chat_boundary_info(&self) -> Result<ChatBoundaryInfo> {
        let eos_text = self.get_eos_text().unwrap_or_default();
        probe_chat_boundary_info(
            |messages_json, add_assistant| {
                self.apply_chat_template_json(messages_json, add_assistant)
            },
            eos_text,
        )
    }
}

fn template_message(role: &'static str, content: &'static str) -> serde_json::Value {
    json!({
        "role": role,
        "content": content,
    })
}

fn messages_json(messages: &[&serde_json::Value]) -> Result<String> {
    serde_json::to_string(messages)
        .map_err(|error| Error::RuntimeCommand(format!("failed to render chat JSON: {error}")))
}

fn slice_before_sentinel<'a>(source: &'a str, sentinel: &str) -> &'a str {
    source
        .find(sentinel)
        .map(|index| &source[..index])
        .unwrap_or("")
}

fn slice_after_sentinel<'a>(source: &'a str, sentinel: &str) -> &'a str {
    source
        .find(sentinel)
        .map(|index| &source[index + sentinel.len()..])
        .unwrap_or("")
}

fn unique_non_empty(values: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(values.len());
    for value in values {
        if value.is_empty() || out.iter().any(|seen| seen.as_str() == *value) {
            continue;
        }
        out.push((*value).to_string());
    }
    out
}

#[cfg(test)]
#[path = "tests/chat_tests.rs"]
mod chat_tests;
