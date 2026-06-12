use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use crate::client::{SippCancellationHandle, SippCancellationReason, SippRequestContext};

/// Stable reason attached to gateway cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayCancellationReason {
    /// The downstream client disconnected.
    ClientDisconnected,
    /// The hosting application is shutting down.
    ServerShutdown,
    /// The application explicitly cancelled the request.
    CallerCancelled,
    /// The application deadline expired.
    DeadlineExceeded,
}

impl GatewayCancellationReason {
    const fn into_client(self) -> SippCancellationReason {
        match self {
            Self::ClientDisconnected => SippCancellationReason::ClientDisconnected,
            Self::ServerShutdown => SippCancellationReason::ServerShutdown,
            Self::CallerCancelled => SippCancellationReason::CallerCancelled,
            Self::DeadlineExceeded => SippCancellationReason::DeadlineExceeded,
        }
    }
}

#[derive(Default)]
struct CancellationState {
    reason: Option<GatewayCancellationReason>,
    runs: Vec<SippCancellationHandle>,
}

/// Cancellation source shared by application adapters and execution.
#[derive(Clone, Default)]
pub struct GatewayCancellation {
    state: Arc<Mutex<CancellationState>>,
}

impl GatewayCancellation {
    /// Cancel every client run registered with this request.
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

    /// Return the first cancellation reason.
    pub fn reason(&self) -> Option<GatewayCancellationReason> {
        self.state.lock().ok().and_then(|state| state.reason)
    }

    pub(crate) fn register(&self, handle: SippCancellationHandle) {
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

/// Request metadata propagated through application policy and execution.
#[derive(Clone, Default)]
pub struct GatewayRequestContext {
    /// Application-assigned request identifier.
    pub request_id: Option<String>,
    /// Application-defined metadata available to policy implementations.
    pub metadata: BTreeMap<String, serde_json::Value>,
    /// Request cancellation source.
    pub cancellation: GatewayCancellation,
}

impl GatewayRequestContext {
    /// Create a context for an optional application request identifier.
    pub fn new(request_id: Option<String>) -> Self {
        Self {
            request_id,
            ..Self::default()
        }
    }

    /// Attach application-defined metadata.
    pub fn with_metadata(mut self, metadata: BTreeMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub(crate) fn client_context(&self) -> SippRequestContext {
        SippRequestContext {
            request_id: self.request_id.clone(),
        }
    }
}
