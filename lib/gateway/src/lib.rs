//! Developer-facing gateway protocol helpers.
//!
//! [`sipp::gateway_core`] remains protocol neutral. This package contains
//! route-free HTTP helpers and the optional first-party gateway JSON/SSE
//! profile. Applications own their route handlers.

mod profile;
mod toolkit;

pub use sipp::{
    GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy,
};
pub use profile::GatewayCodec;
pub use toolkit::{
    request_context, request_id, AuthenticatedRequest, Authenticator, DecodedRequest,
    DefaultErrorTranslator, ErrorTranslator, GatewayHttpError, NoAuthentication, ProtocolCodec,
    ToolkitResult,
};
