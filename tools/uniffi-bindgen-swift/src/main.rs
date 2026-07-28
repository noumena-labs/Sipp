//! Swift-specific UniFFI binding generator used by `cargo xtask build swift`.

fn main() {
    uniffi::uniffi_bindgen_swift();
}
