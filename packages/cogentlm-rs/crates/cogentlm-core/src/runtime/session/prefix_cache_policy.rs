use crate::runtime::llama_token;

pub const PREFIX_HASH_SEED: u64 = 1_469_598_103_934_665_603;
pub const PREFIX_HASH_PRIME: u64 = 1_099_511_628_211;

pub fn mix_prefix_hash_token(hash: u64, token: llama_token) -> u64 {
    (hash ^ (token as u32 as u64)).wrapping_mul(PREFIX_HASH_PRIME)
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
        token_count % self.prefix_cache_interval_tokens == 0
    }

    pub fn build_candidate_boundaries(&self, tokens: &[llama_token]) -> Vec<PrefixCacheBoundary> {
        if tokens.len() < self.minimum_prefix_cache_tokens {
            return Vec::new();
        }

        let mut boundaries = Vec::with_capacity(if self.prefix_cache_interval_tokens == 0 {
            1
        } else {
            tokens.len() / self.prefix_cache_interval_tokens + 1
        });
        let mut rolling_hash = PREFIX_HASH_SEED;
        for (index, token) in tokens.iter().enumerate() {
            rolling_hash = mix_prefix_hash_token(rolling_hash, *token);
            let token_count = index + 1;
            if self.should_store_boundary(token_count, tokens.len()) {
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
        self.stats.lookup_count += 1;
    }

    pub fn record_hit(&mut self, token_count: usize) {
        self.stats.hit_count += 1;
        self.stats.restored_token_count += token_count as u64;
    }

    pub fn record_store(&mut self, token_count: usize) {
        self.stats.store_count += 1;
        self.stats.stored_token_count += token_count as u64;
    }

    pub fn stats(&self) -> PrefixCachePolicyStats {
        self.stats
    }
}

impl Default for PrefixCachePolicy {
    fn default() -> Self {
        Self::new(128)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_zero_stores_only_terminal_boundaries() {
        let policy = PrefixCachePolicy::new(0);
        assert!(!policy.should_store_boundary(31, 64));
        assert!(!policy.should_store_boundary(32, 64));
        assert!(policy.should_store_boundary(64, 64));
    }

    #[test]
    fn candidate_boundaries_are_longest_first() {
        let policy = PrefixCachePolicy::new(4);
        let tokens: Vec<_> = (0..10).collect();
        let boundaries = policy.build_candidate_boundaries(&tokens);
        let counts: Vec<_> = boundaries
            .iter()
            .map(|boundary| boundary.token_count)
            .collect();
        assert_eq!(counts, vec![10, 8, 4]);
    }

    #[test]
    fn hash_prefix_clamps_to_available_tokens() {
        let policy = PrefixCachePolicy::new(4);
        let tokens = vec![1, 2, 3];
        assert_eq!(
            policy.hash_prefix(&tokens, 99),
            policy.hash_prefix(&tokens, 3)
        );
        assert_ne!(policy.hash_prefix(&tokens, 2), 0);
    }
}
