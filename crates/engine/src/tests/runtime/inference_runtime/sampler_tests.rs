//! Tests the `runtime::inference_runtime::sampler` module in
//! `cogentlm-engine`.
//!
//! Covers sampler attach/detach, resident backend sampler parking, and CPU
//! sampler pooling with empty sampler handles and no native sampler creation.

use std::collections::HashMap;

use crate::error::Error;
use crate::native_bridge::{NativeRuntimeHandle, SamplerHandle};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::scheduler::{SamplerCacheKey, SlotPhase, SlotScheduler, SlotState};

use super::*;

fn key(name: &str) -> SamplerCacheKey {
    SamplerCacheKey {
        sampling_json: format!(r#"{{"name":"{name}"}}"#),
        grammar: String::new(),
        json_schema: String::new(),
    }
}

#[test]
fn attach_backend_sampler_rejects_ineligible_slots_without_mutation() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = SlotState::new(0);

    assert!(!attach_backend_sampler(&mut runtime, &mut slot));
    assert!(!slot.backend_sampler_attached);

    slot.seq_id = 0;
    slot.set_sampler(SamplerHandle::empty_for_tests());
    assert!(!attach_backend_sampler(&mut runtime, &mut slot));
    assert!(!slot.backend_sampler_attached);

    slot.backend_sampler_attached = true;
    assert!(!attach_backend_sampler(&mut runtime, &mut slot));
}

#[test]
fn detach_backend_sampler_is_idempotent_and_clears_attached_flag() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let mut slot = SlotState::new(0);

    detach_backend_sampler(&mut runtime, &mut slot);
    assert!(!slot.backend_sampler_attached);

    slot.seq_id = -1;
    slot.backend_sampler_attached = true;
    detach_backend_sampler(&mut runtime, &mut slot);
    assert!(!slot.backend_sampler_attached);
}

#[test]
fn create_sampler_reports_not_ready_for_empty_runtime_handle() {
    let runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();

    assert!(matches!(
        create_sampler(&runtime, &config, None, Some("root ::= \"x\""), None),
        Err(Error::RuntimeNotReady)
    ));
}

#[test]
fn settle_terminal_samplers_pools_non_backend_terminal_slots_only() {
    let pooled_key = key("pooled");
    let active_key = key("active");
    let mut scheduler = SlotScheduler::default();
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();

    let mut completed = SlotState::new(0);
    completed.phase = SlotPhase::Completed;
    completed.set_sampler(SamplerHandle::empty_for_tests());
    completed.sampler_key = Some(pooled_key.clone());

    let mut failed = SlotState::new(1);
    failed.phase = SlotPhase::Failed;
    failed.set_sampler(SamplerHandle::empty_for_tests());
    failed.sampler_key = Some(pooled_key.clone());

    let mut active = SlotState::new(2);
    active.phase = SlotPhase::Decode;
    active.set_sampler(SamplerHandle::empty_for_tests());
    active.sampler_key = Some(active_key.clone());

    scheduler.slots.push(completed);
    scheduler.slots.push(failed);
    scheduler.slots.push(active);

    let mut pool = HashMap::new();
    let mut resident = HashMap::new();
    settle_terminal_samplers(
        &mut scheduler,
        &mut native_runtime,
        &mut pool,
        &mut resident,
    );

    assert_eq!(pool.get(&pooled_key).map(Vec::len), Some(2));
    assert!(resident.is_empty());
    assert!(scheduler.slots[0].sampler.is_none());
    assert!(scheduler.slots[1].sampler.is_none());
    assert!(scheduler.slots[2].sampler.is_some());
    assert_eq!(scheduler.slots[2].sampler_key, Some(active_key));
}

#[test]
fn settle_terminal_samplers_drops_non_backend_sampler_without_cache_key() {
    let mut scheduler = SlotScheduler::default();
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();
    let mut completed = SlotState::new(0);
    completed.phase = SlotPhase::Completed;
    completed.set_sampler(SamplerHandle::empty_for_tests());

    scheduler.slots.push(completed);

    let mut pool = HashMap::new();
    let mut resident = HashMap::new();
    settle_terminal_samplers(
        &mut scheduler,
        &mut native_runtime,
        &mut pool,
        &mut resident,
    );

    assert!(pool.is_empty());
    assert!(resident.is_empty());
    assert!(scheduler.slots[0].sampler.is_none());
}

#[test]
fn settle_terminal_samplers_parks_completed_backend_sampler_by_sequence() {
    let pooled_key = key("pooled");
    let mut scheduler = SlotScheduler::default();
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();

    let mut completed = SlotState::new(0);
    completed.seq_id = 10;
    completed.phase = SlotPhase::Completed;
    completed.set_sampler(SamplerHandle::empty_for_tests());
    completed.backend_sampler_attached = true;
    completed.sampler_key = Some(pooled_key.clone());

    scheduler.slots.push(completed);

    let mut pool = HashMap::new();
    let mut resident = HashMap::new();
    settle_terminal_samplers(
        &mut scheduler,
        &mut native_runtime,
        &mut pool,
        &mut resident,
    );

    assert!(pool.is_empty());
    assert_eq!(
        resident.get(&10).map(|sampler| &sampler.key),
        Some(&pooled_key)
    );
    assert!(scheduler.slots[0].sampler.is_none());
    assert!(scheduler.slots[0].sampler_key.is_none());
    assert!(!scheduler.slots[0].backend_sampler_attached);
}

#[test]
fn settle_terminal_samplers_drops_failed_backend_sampler_without_pooling() {
    let pooled_key = key("pooled");
    let mut scheduler = SlotScheduler::default();
    let mut native_runtime = NativeRuntimeHandle::empty_for_tests();

    let mut failed = SlotState::new(0);
    failed.seq_id = 10;
    failed.phase = SlotPhase::Failed;
    failed.set_sampler(SamplerHandle::empty_for_tests());
    failed.backend_sampler_attached = true;
    failed.sampler_key = Some(pooled_key);

    scheduler.slots.push(failed);

    let mut pool = HashMap::new();
    let mut resident = HashMap::new();
    settle_terminal_samplers(
        &mut scheduler,
        &mut native_runtime,
        &mut pool,
        &mut resident,
    );

    assert!(pool.is_empty());
    assert!(resident.is_empty());
    assert!(scheduler.slots[0].sampler.is_none());
    assert!(scheduler.slots[0].sampler_key.is_none());
    assert!(!scheduler.slots[0].backend_sampler_attached);
}

#[test]
fn runtime_sampler_cleanup_methods_cover_resident_and_active_paths() {
    let pooled_key = key("pooled");
    let mut runtime = test_runtime(NativeRuntimeConfig::default());

    let mut completed = SlotState::new(0);
    completed.seq_id = 10;
    completed.phase = SlotPhase::Completed;
    completed.set_sampler(SamplerHandle::empty_for_tests());
    completed.backend_sampler_attached = true;
    completed.sampler_key = Some(pooled_key.clone());

    let mut active = SlotState::new(1);
    active.seq_id = 11;
    active.phase = SlotPhase::Decode;
    active.backend_sampler_attached = true;

    runtime.slot_scheduler.slots.push(completed);
    runtime.slot_scheduler.slots.push(active);

    runtime.settle_terminal_samplers_locked();
    assert!(!runtime.slot_scheduler.slots[0].backend_sampler_attached);
    assert!(runtime.slot_scheduler.slots[1].backend_sampler_attached);
    assert_eq!(
        runtime
            .resident_backend_samplers
            .get(&10)
            .map(|sampler| &sampler.key),
        Some(&pooled_key)
    );
    assert!(runtime.sampler_pool.is_empty());

    runtime.detach_all_backend_samplers_locked();
    assert!(!runtime.slot_scheduler.slots[1].backend_sampler_attached);
    assert!(runtime.resident_backend_samplers.is_empty());
}
