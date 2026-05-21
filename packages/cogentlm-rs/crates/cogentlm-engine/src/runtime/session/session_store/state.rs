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

impl SequenceState {
    pub fn in_sync(&self) -> bool {
        usize::try_from(self.n_past).is_ok_and(|n_past| n_past == self.current_kv_tokens.len())
    }
}

#[cfg(test)]
mod tests {
    use super::SequenceState;

    #[test]
    fn sequence_state_reports_sync_with_kv_length() {
        let mut state = SequenceState {
            current_kv_tokens: vec![1, 2, 3],
            n_past: 3,
            ..SequenceState::default()
        };
        assert!(state.in_sync());

        state.n_past = 2;
        assert!(!state.in_sync());

        state.n_past = -1;
        assert!(!state.in_sync());
    }
}
