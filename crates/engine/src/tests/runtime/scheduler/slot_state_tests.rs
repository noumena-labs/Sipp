//! Unit tests for the parent module.

use super::*;
use crate::runtime::session::{CacheCandidate, KvCacheAdmission, SequenceMirror};

fn admission(seq_id: i32, generation: u64, tokens: Vec<i32>) -> KvCacheAdmission {
    KvCacheAdmission {
        seq_id,
        generation,
        mirror: SequenceMirror {
            n_past: tokens.len() as i32,
            current_kv_tokens: tokens,
        },
        candidate: CacheCandidate::Live,
    }
}

#[test]
fn attach_request_copies_session_mirror_and_resets_slot_progress() {
    let mut slot = SlotState::new(3);
    slot.prefill_cursor = 9;
    slot.generated_tokens.push(99);
    slot.sampler_key = Some(SamplerCacheKey {
        sampling_json: "sampling".to_string(),
        grammar: String::new(),
        json_schema: String::new(),
    });
    let mut request = GenerateRequest::new(42, "ctx");
    request.prompt_tokens = vec![1, 2, 3];

    slot.attach_request(request, admission(7, 11, vec![1]));

    assert_eq!(slot.slot_id, 3);
    assert_eq!(slot.request_id, 42);
    assert_eq!(slot.seq_id, 7);
    assert_eq!(slot.lease_generation, 11);
    assert_eq!(slot.cache_candidate, CacheCandidate::Live);
    assert_eq!(slot.phase, SlotPhase::Admitted);
    assert_eq!(slot.prefill_cursor, 0);
    assert!(!slot.sampler_prompt_seeded);
    assert!(slot.sampler_key.is_none());
    assert!(slot.generated_tokens.is_empty());
    assert_eq!(slot.mirror.current_kv_tokens, vec![1]);
    assert_eq!(slot.mirror.n_past, 1);
}

#[test]
fn reset_to_idle_clears_request_and_runtime_buffers() {
    let mut slot = SlotState::new(1);
    slot.attach_request(GenerateRequest::new(2, "ctx"), KvCacheAdmission::default());
    slot.buffered_output_text = "abc".to_string();
    slot.pending_emission_text = "def".to_string();
    slot.pending_utf8_bytes = b"ghi".to_vec();

    slot.reset_to_idle();

    assert_eq!(slot.phase, SlotPhase::Idle);
    assert_eq!(slot.seq_id, -1);
    assert_eq!(slot.lease_generation, 0);
    assert_eq!(slot.cache_candidate, CacheCandidate::None);
    assert_eq!(slot.request_id, 0);
    assert!(slot.request.is_none());
    assert!(slot.buffered_output_text.is_empty());
    assert!(slot.pending_emission_text.is_empty());
    assert!(slot.pending_utf8_bytes.is_empty());
}
