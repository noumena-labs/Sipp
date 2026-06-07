use std::sync::{Arc, Mutex};

use cogentlm_client::{CogentCancellationHandle, CogentCancellationReason, CogentRequestContext};

use crate::{GatewayCaller, GatewayError, GatewayErrorKind, GatewayResult};

/// Maximum accepted request ID length.
pub const MAX_REQUEST_ID_BYTES: usize = 128;

/// Stable reason attached to gateway cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayCancellationReason {
    /// The downstream HTTP client disconnected.
    ClientDisconnected,
    /// The hosting server is shutting down.
    ServerShutdown,
    /// The application explicitly cancelled the request.
    CallerCancelled,
    /// The request exceeded its deadline.
    DeadlineExceeded,
}

impl GatewayCancellationReason {
    /// Stable label used by logs and metrics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClientDisconnected => "client_disconnected",
            Self::ServerShutdown => "server_shutdown",
            Self::CallerCancelled => "caller_cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
        }
    }

    pub(crate) const fn into_client(self) -> CogentCancellationReason {
        match self {
            Self::ClientDisconnected => CogentCancellationReason::ClientDisconnected,
            Self::ServerShutdown => CogentCancellationReason::ServerShutdown,
            Self::CallerCancelled => CogentCancellationReason::CallerCancelled,
            Self::DeadlineExceeded => CogentCancellationReason::DeadlineExceeded,
        }
    }
}

#[derive(Default)]
struct CancellationState {
    reason: Option<GatewayCancellationReason>,
    runs: Vec<CogentCancellationHandle>,
}

/// Cloneable cancellation source shared by adapters and host frameworks.
#[derive(Clone, Default)]
pub struct GatewayCancellation {
    state: Arc<Mutex<CancellationState>>,
}

impl GatewayCancellation {
    /// Cancel the request and every registered client run.
    pub fn cancel(&self, reason: GatewayCancellationReason) {
        let handles = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            if state.reason.is_some() {
                return;
            }
            state.reason = Some(reason);
            std::mem::take(&mut state.runs)
        };
        for handle in handles {
            handle.cancel(reason.into_client());
        }
    }

    /// Return the first cancellation reason, if any.
    pub fn reason(&self) -> Option<GatewayCancellationReason> {
        self.state.lock().ok().and_then(|state| state.reason)
    }

    pub(crate) fn register(&self, handle: CogentCancellationHandle) {
        let reason = {
            let Ok(mut state) = self.state.lock() else {
                return;
            };
            match state.reason {
                Some(reason) => Some(reason),
                None => {
                    state.runs.push(handle.clone());
                    None
                }
            }
        };
        if let Some(reason) = reason {
            handle.cancel(reason.into_client());
        }
    }
}

/// Request metadata propagated through the gateway and client layers.
#[derive(Clone)]
pub struct GatewayRequestContext {
    /// Canonical request ID assigned at the HTTP or framework boundary.
    pub request_id: String,
    /// Authenticated caller and access scope.
    pub caller: GatewayCaller,
    /// Request cancellation source.
    pub cancellation: GatewayCancellation,
}

impl GatewayRequestContext {
    /// Create a validated request context.
    pub fn new(request_id: impl Into<String>, caller: GatewayCaller) -> GatewayResult<Self> {
        let request_id = request_id.into();
        validate_request_id(&request_id)?;
        Ok(Self {
            request_id,
            caller,
            cancellation: GatewayCancellation::default(),
        })
    }

    /// Create a context with an existing cancellation source.
    pub fn with_cancellation(
        request_id: impl Into<String>,
        caller: GatewayCaller,
        cancellation: GatewayCancellation,
    ) -> GatewayResult<Self> {
        let mut context = Self::new(request_id, caller)?;
        context.cancellation = cancellation;
        Ok(context)
    }

    pub(crate) fn client_context(&self) -> CogentRequestContext {
        CogentRequestContext {
            request_id: Some(self.request_id.clone()),
        }
    }
}

/// Validate a caller-provided request ID.
pub fn validate_request_id(request_id: &str) -> GatewayResult<()> {
    if request_id.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "request ID must not be empty",
        ));
    }
    if request_id.len() > MAX_REQUEST_ID_BYTES {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "request ID exceeds the maximum length",
        ));
    }
    if !request_id.is_ascii()
        || request_id
            .bytes()
            .any(|byte| byte.is_ascii_control() || byte.is_ascii_whitespace())
    {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "request ID must be visible ASCII without whitespace",
        ));
    }
    Ok(())
}
