use crate::runtime::{llama_seq_id, llama_token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceState {
    pub current_kv_tokens: Vec<llama_token>,
    pub n_past: i32,
    pub hardware_id: llama_seq_id,
    pub pin_count: usize,
}

impl Default for SequenceState {
    fn default() -> Self {
        Self {
            current_kv_tokens: Vec::new(),
            n_past: 0,
            hardware_id: -1,
            pin_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    mod state_tests;
}
