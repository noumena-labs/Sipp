//! Protocol-neutral gateway execution primitives.
//!
//! Applications supply target resolution, authorization, admission control,
//! and protocol adapters. This crate only orders those decisions around typed
//! query, chat, and embed execution.

mod context;
mod error;
mod pipeline;

pub use context::{GatewayCancellation, GatewayCancellationReason, GatewayRequestContext};
pub use error::{GatewayError, GatewayErrorKind, GatewayResult};
pub use pipeline::{
    AdmissionController, AdmissionPermit, AllowAllAuthorizer, Authorizer, GatewayExecutor,
    GatewayPipeline, GatewayStream, GatewayStreamEvent, Operation, SippClientExecutor,
    TargetResolver, UnlimitedAdmissionController,
};
