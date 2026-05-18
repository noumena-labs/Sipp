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
    let mut capacity = i32::try_from(text.len().saturating_add(8)).map_err(|_| Error::Tokenize)?;
    capacity = capacity.max(8);

    loop {
        let mut tokens = vec![0; capacity as usize];
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
            tokens.truncate(result as usize);
            return Ok(tokens);
        }

        if result == i32::MIN {
            return Err(Error::Tokenize);
        }

        let needed = result.saturating_abs();
        if needed <= capacity {
            return Err(Error::Tokenize);
        }
        capacity = needed;
    }
}

pub fn token_to_piece(
    vocab: NonNull<ffi::llama_vocab>,
    token: ffi::llama_token,
    special: bool,
) -> Result<String> {
    let mut capacity = 64_i32;

    loop {
        let mut buffer = vec![0 as c_char; capacity as usize];
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
            let bytes =
                unsafe { slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), result as usize) };
            return Ok(String::from_utf8_lossy(bytes).into_owned());
        }

        if result == 0 || result == i32::MIN {
            return Err(Error::TokenToPiece { token });
        }

        let needed = result.saturating_abs();
        if needed <= capacity {
            return Err(Error::TokenToPiece { token });
        }
        capacity = needed;
    }
}
