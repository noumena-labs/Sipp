use serde::Serialize;

pub(crate) fn serialize_json_response<T>(response: &T) -> String
where
    T: Serialize,
{
    serde_json::to_string(response).unwrap_or_else(|_| {
        "{\"ok\":false,\"error\":{\"code\":\"SERIALIZATION_FAILED\",\"message\":\"failed to \
         serialize browser FFI response\"}}"
            .to_string()
    })
}

#[cfg(test)]
#[path = "tests/ffi_tests.rs"]
mod ffi_tests;
