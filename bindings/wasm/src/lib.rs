#![cfg_attr(not(target_family = "wasm"), allow(dead_code))]

mod abi;
pub mod engine;
#[cfg(target_family = "wasm")]
mod exports;
mod ffi;
pub mod gguf;
pub mod hash;
pub mod ingest;
#[cfg(target_family = "wasm")]
pub mod lifecycle;
#[cfg(target_family = "wasm")]
pub mod pairing;

pub use abi::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};
pub use engine::*;
pub use hash::*;
