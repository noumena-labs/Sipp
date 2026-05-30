//! Provider-neutral public value types shared by the local engine and provider layer.
//! This crate intentionally has no runtime, async, or HTTP dependencies.

mod capability;
mod chat;
mod result;
mod stream;
mod token_usage;

pub use capability::CapabilitySupport;
pub use chat::{ChatMessage, ChatRole};
pub use result::FinishReason;
pub use stream::{StreamStats, TokenBatch};
pub use token_usage::TokenUsage;
