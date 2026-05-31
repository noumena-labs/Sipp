//! Provider-neutral public value types shared by the local engine and provider layer.
//! This crate intentionally has no runtime, async, or HTTP dependencies.

mod capability;
mod chat;
mod result;
mod token_emission;
mod token_usage;

pub use capability::CapabilitySupport;
pub use chat::{ChatMessage, ChatRole};
pub use result::FinishReason;
pub use token_emission::{TokenBatch, TokenEmissionStats};
pub use token_usage::TokenUsage;
