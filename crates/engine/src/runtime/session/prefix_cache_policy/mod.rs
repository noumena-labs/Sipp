//! Boundary picker for snapshot prefix-cache entries. Decides where along a prompt to commit a KV snapshot.

use crate::runtime::llama_token;
use crate::runtime::numeric::saturating_usize_to_u64;

pub const PREFIX_HASH_SEED: u64 = 1_469_598_103_934_665_603;
pub const PREFIX_HASH_PRIME: u64 = 1_099_511_628_211;
const DEFAULT_PREFIX_CACHE_INTERVAL_TOKENS: usize = 128;
const MAX_MINIMUM_PREFIX_CACHE_TOKENS: usize = 32;

pub fn mix_prefix_hash_token(hash: u64, token: llama_token) -> u64 {
    let token_bits = u32::from_ne_bytes(token.to_ne_bytes());
    (hash ^ u64::from(token_bits)).wrapping_mul(PREFIX_HASH_PRIME)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefixCacheBoundary {
    pub token_count: usize,
    pub prefix_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrefixCachePolicyStats {
    pub lookup_count: u64,
    pub hit_count: u64,
    pub store_count: u64,
    pub restored_token_count: u64,
    pub stored_token_count: u64,
}

impl PrefixCachePolicyStats {
    fn record_lookup(&mut self) {
        increment_counter(&mut self.lookup_count);
    }

    fn record_hit(&mut self, token_count: usize) {
        increment_counter(&mut self.hit_count);
        add_token_count(&mut self.restored_token_count, token_count);
    }

    fn record_store(&mut self, token_count: usize) {
        increment_counter(&mut self.store_count);
        add_token_count(&mut self.stored_token_count, token_count);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCachePolicy {
    prefix_cache_interval_tokens: usize,
    minimum_prefix_cache_tokens: usize,
    pub(crate) stats: PrefixCachePolicyStats,
}

impl PrefixCachePolicy {
    pub fn new(prefix_cache_interval_tokens: usize) -> Self {
        Self {
            prefix_cache_interval_tokens,
            minimum_prefix_cache_tokens: minimum_prefix_cache_tokens(prefix_cache_interval_tokens),
            stats: PrefixCachePolicyStats::default(),
        }
    }

    pub fn should_store_boundary(&self, token_count: usize, terminal_token_count: usize) -> bool {
        if token_count < self.minimum_prefix_cache_tokens {
            return false;
        }
        if is_terminal_boundary(token_count, terminal_token_count) {
            return true;
        }
        is_interval_boundary(token_count, self.prefix_cache_interval_tokens)
    }

    pub fn build_candidate_boundaries(&self, tokens: &[llama_token]) -> Vec<PrefixCacheBoundary> {
        let len = tokens.len();
        if len < self.minimum_prefix_cache_tokens {
            return Vec::new();
        }

        let interval = self.prefix_cache_interval_tokens;

        if interval == 0 {
            return vec![PrefixCacheBoundary {
                token_count: len,
                prefix_hash: hash_tokens(tokens),
            }];
        }

        let mut boundaries = Vec::with_capacity(candidate_boundary_capacity(len, interval));
        let mut rolling_hash = PREFIX_HASH_SEED;

        for (index, &token) in tokens.iter().enumerate() {
            rolling_hash = mix_prefix_hash_token(rolling_hash, token);
            let token_count = index + 1;
            if self.should_store_boundary(token_count, len) {
                boundaries.push(PrefixCacheBoundary {
                    token_count,
                    prefix_hash: rolling_hash,
                });
            }
        }
        boundaries.reverse();
        boundaries
    }

    pub fn hash_prefix(&self, tokens: &[llama_token], token_count: usize) -> u64 {
        if token_count == 0 || tokens.is_empty() {
            return 0;
        }

        hash_tokens(&tokens[..token_count.min(tokens.len())])
    }

    pub fn record_lookup(&mut self) {
        self.stats.record_lookup();
    }

    pub fn record_hit(&mut self, token_count: usize) {
        self.stats.record_hit(token_count);
    }

    pub fn record_store(&mut self, token_count: usize) {
        self.stats.record_store(token_count);
    }
}

impl Default for PrefixCachePolicy {
    fn default() -> Self {
        Self::new(DEFAULT_PREFIX_CACHE_INTERVAL_TOKENS)
    }
}

fn hash_tokens(tokens: &[llama_token]) -> u64 {
    tokens.iter().fold(PREFIX_HASH_SEED, |hash, &token| {
        mix_prefix_hash_token(hash, token)
    })
}

fn minimum_prefix_cache_tokens(prefix_cache_interval_tokens: usize) -> usize {
    if prefix_cache_interval_tokens == 0 {
        MAX_MINIMUM_PREFIX_CACHE_TOKENS
    } else {
        prefix_cache_interval_tokens.min(MAX_MINIMUM_PREFIX_CACHE_TOKENS)
    }
}

fn candidate_boundary_capacity(token_count: usize, interval: usize) -> usize {
    token_count
        .checked_div(interval)
        .map(|capacity| capacity + 1)
        .unwrap_or(1)
}

fn is_terminal_boundary(token_count: usize, terminal_token_count: usize) -> bool {
    token_count == terminal_token_count
}

fn is_interval_boundary(token_count: usize, interval: usize) -> bool {
    interval > 0 && token_count.is_multiple_of(interval)
}

fn increment_counter(counter: &mut u64) {
    *counter = counter.saturating_add(1);
}

fn add_token_count(total: &mut u64, token_count: usize) {
    *total = total.saturating_add(saturating_usize_to_u64(token_count));
}

#[cfg(test)]
mod tests {
    mod prefix_cache_policy_tests;
}
