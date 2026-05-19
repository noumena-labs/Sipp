use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::lifecycle::util::hex_lower;
use crate::lifecycle::ModelError;

use super::COPY_BUFFER_BYTES;

const INSPECTION_PREFIX_BYTES: usize = 8 * 1024 * 1024;

pub(super) fn inspect_local_path(source_path: &Path) -> Result<(String, Vec<u8>), ModelError> {
    let mut source = File::open(source_path)?;
    let prefix_capacity = source
        .metadata()
        .ok()
        .map(|metadata| inspection_prefix_capacity(metadata.len()))
        .unwrap_or(INSPECTION_PREFIX_BYTES);
    let mut hasher = Sha256::new();
    let mut prefix = Vec::with_capacity(prefix_capacity);
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];

    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        if prefix.len() < INSPECTION_PREFIX_BYTES {
            let remaining = INSPECTION_PREFIX_BYTES - prefix.len();
            prefix.extend_from_slice(&buffer[..read.min(remaining)]);
        }
    }

    Ok((hex_lower(&hasher.finalize()), prefix))
}

fn inspection_prefix_capacity(source_len: u64) -> usize {
    usize::try_from(source_len)
        .ok()
        .map_or(INSPECTION_PREFIX_BYTES, |len| {
            len.min(INSPECTION_PREFIX_BYTES)
        })
}

pub(crate) fn hash_file(source_path: &Path) -> Result<String, ModelError> {
    let mut source = File::open(source_path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];

    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex_lower(&hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspection_prefix_capacity_clamps_to_prefix_limit() {
        assert_eq!(inspection_prefix_capacity(0), 0);
        assert_eq!(inspection_prefix_capacity(1024), 1024);
        assert_eq!(
            inspection_prefix_capacity((INSPECTION_PREFIX_BYTES as u64) + 1),
            INSPECTION_PREFIX_BYTES
        );
        assert_eq!(
            inspection_prefix_capacity(u64::MAX),
            INSPECTION_PREFIX_BYTES
        );
    }
}
