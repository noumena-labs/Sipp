use crate::runtime::llama_seq_id;

use super::SessionStore;

impl SessionStore {
    pub fn acquire_seq_id(&mut self, hint: llama_seq_id) -> llama_seq_id {
        if let Some(hint_index) = seq_id_index(hint, self.seq_id_available.len()) {
            if let Some(index) = self
                .free_seq_ids
                .iter()
                .position(|candidate| *candidate == hint)
            {
                self.free_seq_ids.remove(index);
                self.seq_id_available[hint_index] = false;
                return hint;
            }
        }

        let Some(seq_id) = self.free_seq_ids.pop_front() else {
            return -1;
        };
        if let Some(seq_index) = seq_id_index(seq_id, self.seq_id_available.len()) {
            self.seq_id_available[seq_index] = false;
        }

        seq_id
    }

    pub fn release_seq_id(&mut self, seq_id: llama_seq_id) {
        let Some(seq_index) = seq_id_index(seq_id, self.seq_id_available.len()) else {
            return;
        };
        if self.seq_id_available[seq_index] {
            return;
        }

        self.seq_id_available[seq_index] = true;
        // Keep serial runs on the warm physical KV sequence.
        self.free_seq_ids.push_front(seq_id);
    }
}

fn seq_id_index(seq_id: llama_seq_id, len: usize) -> Option<usize> {
    let index = usize::try_from(seq_id).ok()?;
    (index < len).then_some(index)
}

pub(super) fn clamp_sequence_capacity(max_sequences: usize) -> usize {
    max_sequences.clamp(1, max_representable_sequences())
}

fn max_representable_sequences() -> usize {
    usize::try_from(llama_seq_id::MAX)
        .ok()
        .and_then(|value| value.checked_add(1))
        .unwrap_or(usize::MAX)
}

#[cfg(test)]
mod tests {
    mod sequence_ids_tests;
}
