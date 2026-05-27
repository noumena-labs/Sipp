use std::fs::File;
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::defaults::BYTES_PER_MIB;
use crate::lifecycle::util::hex_lower;
use crate::lifecycle::ModelError;

use super::COPY_BUFFER_BYTES;

const INSPECTION_PREFIX_BYTES: usize = 8 * BYTES_PER_MIB;

pub(super) fn inspect_local_path(source_path: &Path) -> Result<(String, Vec<u8>), ModelError> {
    let mut source = File::open(source_path)?;
    let prefix_capacity = source
        .metadata()
        .ok()
        .map(|metadata| inspection_prefix_capacity(metadata.len()))
        .unwrap_or(INSPECTION_PREFIX_BYTES);
    hash_reader(&mut source, prefix_capacity)
}

fn hash_reader(
    source: &mut impl Read,
    prefix_limit: usize,
) -> Result<(String, Vec<u8>), ModelError> {
    let mut hasher = Sha256::new();
    let mut prefix = Vec::with_capacity(prefix_limit);
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];

    loop {
        let read = source.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        append_prefix_bytes(&mut prefix, &buffer[..read], prefix_limit);
    }

    Ok((hex_lower(&hasher.finalize()), prefix))
}

fn append_prefix_bytes(prefix: &mut Vec<u8>, bytes: &[u8], prefix_limit: usize) {
    if prefix.len() >= prefix_limit {
        return;
    }
    let remaining = prefix_limit - prefix.len();
    prefix.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
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
    hash_reader(&mut source, 0).map(|(hash, _)| hash)
}

#[cfg(test)]
mod tests {
    mod content_tests;
}
