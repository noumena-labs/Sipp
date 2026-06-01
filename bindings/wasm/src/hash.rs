use sha2::{Digest, Sha256};

pub struct BrowserSha256Hasher {
    hasher: Sha256,
}

impl BrowserSha256Hasher {
    pub(crate) fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }

    pub(crate) fn finalize_hex(self) -> String {
        hex_lower(&self.hasher.finalize())
    }
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    mod hash_tests;
}
