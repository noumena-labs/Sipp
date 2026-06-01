//! Unit tests for the parent module.

use serde::ser::Error;
use serde::Serialize;

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
