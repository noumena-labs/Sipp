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
mod tests {
    use serde::ser::Error;

    use super::*;

    struct Unserializable;

    impl Serialize for Unserializable {
        fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            Err(S::Error::custom("intentional failure"))
        }
    }

    #[test]
    fn serialize_json_response_returns_json_error_on_failure() {
        let response = serialize_json_response(&Unserializable);

        assert!(response.contains("\"ok\":false"));
        assert!(response.contains("\"SERIALIZATION_FAILED\""));
    }
}
