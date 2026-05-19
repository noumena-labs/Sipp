//! Tokenization and detokenization helpers wrapping the llama.cpp vocab.

use std::os::raw::c_char;
use std::ptr::NonNull;
use std::slice;

use cogentlm_sys as ffi;

use crate::error::{Error, Result};

pub fn tokenize(
    vocab: NonNull<ffi::llama_vocab>,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<ffi::llama_token>> {
    let text_len = i32::try_from(text.len()).map_err(|_| Error::Tokenize)?;
    let mut capacity = initial_token_capacity(text.len()).ok_or(Error::Tokenize)?;
    let capacity_usize = usize::try_from(capacity).map_err(|_| Error::Tokenize)?;
    let mut tokens = vec![0; capacity_usize];

    loop {
        // SAFETY: `vocab` is a non-null llama vocabulary pointer owned by the
        // runtime, `text.as_ptr()` is valid for `text_len` bytes, and `tokens`
        // is valid for `capacity` llama_token writes for the duration of the call.
        let result = unsafe {
            ffi::llama_tokenize(
                vocab.as_ptr(),
                text.as_ptr().cast(),
                text_len,
                tokens.as_mut_ptr(),
                capacity,
                add_special,
                parse_special,
            )
        };

        if result >= 0 && result <= capacity {
            let result = usize::try_from(result).map_err(|_| Error::Tokenize)?;
            tokens.truncate(result);
            return Ok(tokens);
        }

        capacity = next_ffi_capacity(result, capacity).ok_or(Error::Tokenize)?;
        tokens.resize(usize::try_from(capacity).map_err(|_| Error::Tokenize)?, 0);
    }
}

pub fn token_to_piece(
    vocab: NonNull<ffi::llama_vocab>,
    token: ffi::llama_token,
    special: bool,
) -> Result<String> {
    let mut capacity = 64_i32;
    let mut buffer = vec![
        c_char::default();
        usize::try_from(capacity).map_err(|_| Error::TokenToPiece { token })?
    ];

    loop {
        // SAFETY: `vocab` is non-null, `buffer` is valid for `capacity`
        // c_char writes, and llama.cpp writes at most the returned byte count.
        let result = unsafe {
            ffi::llama_token_to_piece(
                vocab.as_ptr(),
                token,
                buffer.as_mut_ptr(),
                capacity,
                0,
                special,
            )
        };

        if result >= 0 && result <= capacity {
            let result = usize::try_from(result).map_err(|_| Error::TokenToPiece { token })?;
            // SAFETY: `buffer` remains alive and initialized for `result`
            // bytes. `result <= capacity`, and `capacity` was used to size
            // the allocation for the FFI call above.
            let bytes = unsafe { slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), result) };
            return Ok(String::from_utf8_lossy(bytes).into_owned());
        }

        capacity = next_ffi_capacity(result, capacity).ok_or(Error::TokenToPiece { token })?;
        buffer.resize(
            usize::try_from(capacity).map_err(|_| Error::TokenToPiece { token })?,
            c_char::default(),
        );
    }
}

fn initial_token_capacity(text_len: usize) -> Option<i32> {
    let padded = text_len.checked_add(8)?;
    let capacity = i32::try_from(padded).ok()?;
    Some(capacity.max(8))
}

fn next_ffi_capacity(result: i32, current: i32) -> Option<i32> {
    if result == i32::MIN {
        return None;
    }
    let needed = if result < 0 {
        result.checked_abs()?
    } else {
        result
    };
    (needed > current).then_some(needed)
}

#[cfg(test)]
mod tests;
