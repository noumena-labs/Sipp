//! FFI-facing helpers for runtime initialization, metadata, and shim strings.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};
use crate::runtime::config::ResolvedRuntimeLimits;
use crate::token::token_to_piece;

use super::InferenceRuntime;

pub(super) fn c_strings_from_args(args: &[String]) -> Result<Vec<CString>> {
    args.iter()
        .map(|arg| CString::new(arg.as_str()).map_err(Error::from))
        .collect()
}

pub(super) fn c_ptrs_from_strings(args: &[CString]) -> Vec<*const c_char> {
    args.iter().map(|arg| arg.as_ptr()).collect()
}

pub(super) fn runtime_command_from_shim_error(value: *mut c_char, fallback: &'static str) -> Error {
    if value.is_null() {
        return Error::RuntimeCommand(fallback.to_string());
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Error::RuntimeCommand(result)
}

pub(super) fn owned_shim_string(value: *mut c_char, name: &'static str) -> Result<String> {
    if value.is_null() {
        return Err(Error::NullPointer(name));
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        ffi::cogent_free_string(value);
    }
    Ok(result)
}

pub(super) fn resolved_runtime_limits(
    common_init: *mut ffi::cogent_common_init,
) -> ResolvedRuntimeLimits {
    ResolvedRuntimeLimits {
        n_ctx: unsafe { ffi::cogent_common_init_n_ctx(common_init) }.max(0),
        n_batch: unsafe { ffi::cogent_common_init_n_batch(common_init) }.max(0),
        n_ubatch: unsafe { ffi::cogent_common_init_n_ubatch(common_init) }.max(0),
        n_parallel: unsafe { ffi::cogent_common_init_n_parallel(common_init) }.max(0),
        kv_unified: unsafe { ffi::cogent_common_init_kv_unified(common_init) },
        flash_attention: owned_shim_string(
            unsafe { ffi::cogent_common_init_flash_attention(common_init) },
            "cogent_common_init_flash_attention",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
        cache_type_k: owned_shim_string(
            unsafe { ffi::cogent_common_init_cache_type_k(common_init) },
            "cogent_common_init_cache_type_k",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
        cache_type_v: owned_shim_string(
            unsafe { ffi::cogent_common_init_cache_type_v(common_init) },
            "cogent_common_init_cache_type_v",
        )
        .unwrap_or_else(|_| "unknown".to_string()),
    }
}

impl InferenceRuntime {
    pub(super) fn vocab(&self) -> Result<NonNull<ffi::llama_vocab>> {
        if self.primary_model.is_null() {
            return Err(Error::RuntimeNotReady);
        }
        let vocab =
            unsafe { ffi::llama_model_get_vocab(self.primary_model) as *mut ffi::llama_vocab };
        NonNull::new(vocab).ok_or(Error::NullPointer("llama_model_get_vocab"))
    }

    pub fn get_bos_text(&self) -> Result<String> {
        let vocab = self.vocab()?;
        let bos = unsafe { ffi::llama_vocab_bos(vocab.as_ptr()) };
        if bos == ffi::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        token_to_piece(vocab, bos, true)
    }

    pub fn get_eos_text(&self) -> Result<String> {
        let vocab = self.vocab()?;
        let eos = unsafe { ffi::llama_vocab_eos(vocab.as_ptr()) };
        if eos == ffi::LLAMA_TOKEN_NULL {
            return Ok(String::new());
        }
        token_to_piece(vocab, eos, true)
    }

    pub fn chat_template_source(&self) -> Result<Option<String>> {
        if self.chat_templates.is_null() {
            return Ok(None);
        }
        owned_shim_string(
            unsafe { ffi::cogent_chat_templates_source(self.chat_templates) },
            "cogent_chat_templates_source",
        )
        .map(Some)
    }

    pub fn apply_chat_template_json(
        &self,
        messages_json: &str,
        add_assistant: bool,
    ) -> Result<String> {
        if self.chat_templates.is_null() {
            return Err(Error::NullPointer("cogent_chat_templates_init"));
        }
        let messages_json = CString::new(messages_json)?;
        owned_shim_string(
            unsafe {
                ffi::cogent_apply_chat_template(
                    self.chat_templates,
                    messages_json.as_ptr(),
                    add_assistant,
                )
            },
            "cogent_apply_chat_template",
        )
    }

    pub fn media_marker(&self) -> Result<String> {
        let marker = unsafe { ffi::cogent_mtmd_default_marker() };
        if marker.is_null() {
            return Err(Error::NullPointer("cogent_mtmd_default_marker"));
        }
        Ok(unsafe { CStr::from_ptr(marker) }
            .to_string_lossy()
            .into_owned())
    }
}
