//! Boundary picker for snapshot prefix-cache entries. Decides where along a prompt to commit a KV snapshot.

use crate::runtime::llama_token;

pub const PREFIX_HASH_SEED: u64 = 1_469_598_103_934_665_603;
pub const PREFIX_HASH_PRIME: u64 = 1_099_511_628_211;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixCachePolicy {
    prefix_cache_interval_tokens: usize,
    minimum_prefix_cache_tokens: usize,
    stats: PrefixCachePolicyStats,
}

impl PrefixCachePolicy {
    pub fn new(prefix_cache_interval_tokens: usize) -> Self {
        Self {
            prefix_cache_interval_tokens,
            minimum_prefix_cache_tokens: if prefix_cache_interval_tokens == 0 {
                32
            } else {
                prefix_cache_interval_tokens.min(32)
            },
            stats: PrefixCachePolicyStats::default(),
        }
    }

    pub fn prefix_cache_interval_tokens(&self) -> usize {
        self.prefix_cache_interval_tokens
    }

    pub fn minimum_prefix_cache_tokens(&self) -> usize {
        self.minimum_prefix_cache_tokens
    }

    pub fn should_store_boundary(&self, token_count: usize, terminal_token_count: usize) -> bool {
        if token_count < self.minimum_prefix_cache_tokens {
            return false;
        }
        if token_count == terminal_token_count {
            return true;
        }
        if self.prefix_cache_interval_tokens == 0 {
            return false;
        }
        token_count.is_multiple_of(self.prefix_cache_interval_tokens)
    }

    pub fn build_candidate_boundaries(&self, tokens: &[llama_token]) -> Vec<PrefixCacheBoundary> {
        let len = tokens.len();
        if len < self.minimum_prefix_cache_tokens {
            return Vec::new();
        }

        let interval = self.prefix_cache_interval_tokens;
        let min_tokens = self.minimum_prefix_cache_tokens;

        if interval == 0 {
            let rolling_hash = self.hash_prefix(tokens, len);
            return vec![PrefixCacheBoundary {
                token_count: len,
                prefix_hash: rolling_hash,
            }];
        }

        let capacity = len / interval + 1;
        let mut boundaries = Vec::with_capacity(capacity);
        let mut rolling_hash = PREFIX_HASH_SEED;

        for (index, &token) in tokens.iter().enumerate() {
            rolling_hash = mix_prefix_hash_token(rolling_hash, token);
            let token_count = index + 1;
            if token_count >= min_tokens && (token_count == len || token_count % interval == 0) {
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

        let mut rolling_hash = PREFIX_HASH_SEED;
        for token in tokens.iter().take(token_count.min(tokens.len())) {
            rolling_hash = mix_prefix_hash_token(rolling_hash, *token);
        }
        rolling_hash
    }

    pub fn record_lookup(&mut self) {
        self.stats.lookup_count = self.stats.lookup_count.saturating_add(1);
    }

    pub fn record_hit(&mut self, token_count: usize) {
        self.stats.hit_count = self.stats.hit_count.saturating_add(1);
        self.stats.restored_token_count = self
            .stats
            .restored_token_count
            .saturating_add(saturating_usize_to_u64(token_count));
    }

    pub fn record_store(&mut self, token_count: usize) {
        self.stats.store_count = self.stats.store_count.saturating_add(1);
        self.stats.stored_token_count = self
            .stats
            .stored_token_count
            .saturating_add(saturating_usize_to_u64(token_count));
    }

    pub fn stats(&self) -> PrefixCachePolicyStats {
        self.stats
    }
}

fn saturating_usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

impl Default for PrefixCachePolicy {
    fn default() -> Self {
        Self::new(128)
    }
}

#[cfg(test)]
mod tests {
    mod prefix_cache_policy_tests;
}
