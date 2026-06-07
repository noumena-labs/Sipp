use std::sync::atomic::{AtomicU8, Ordering};

/// Process lifecycle visible through readiness checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ServerLifecycle {
    /// Listeners are bound while endpoints are loading.
    Starting = 0,
    /// Every configured endpoint is loaded and requests are accepted.
    Ready = 1,
    /// Shutdown has started and new requests are rejected.
    Draining = 2,
    /// Endpoint loading or another startup step failed.
    Failed = 3,
}

/// Atomic lifecycle state.
pub struct LifecycleState {
    value: AtomicU8,
}

impl LifecycleState {
    /// Start in the model-loading state.
    pub const fn starting() -> Self {
        Self {
            value: AtomicU8::new(ServerLifecycle::Starting as u8),
        }
    }

    /// Read the current state.
    pub fn get(&self) -> ServerLifecycle {
        match self.value.load(Ordering::Acquire) {
            1 => ServerLifecycle::Ready,
            2 => ServerLifecycle::Draining,
            3 => ServerLifecycle::Failed,
            _ => ServerLifecycle::Starting,
        }
    }

    /// Set the current state.
    pub fn set(&self, state: ServerLifecycle) {
        self.value.store(state as u8, Ordering::Release);
    }

    /// Return whether public inference is currently accepted.
    pub fn is_ready(&self) -> bool {
        self.get() == ServerLifecycle::Ready
    }
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self::starting()
    }
}
