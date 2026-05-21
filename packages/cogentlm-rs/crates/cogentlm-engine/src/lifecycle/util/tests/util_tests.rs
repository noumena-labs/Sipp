//! Unit tests for the parent module.

use super::super::*;

#[test]
fn hex_lower_encodes_lowercase_nibbles() {
    assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
}
