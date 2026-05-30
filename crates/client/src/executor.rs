use std::future::Future;
use std::sync::{mpsc, Arc};
use std::thread;

use crate::{CogentError, CogentResult};

/// Owned provider I/O executor used by provider endpoints.
#[derive(Clone)]
pub struct ProviderExecutor {
    inner: Arc<ProviderRuntimeThread>,
}

struct ProviderRuntimeThread {
    handle: tokio::runtime::Handle,
    // Only `Drop` touches this, and `Arc` gives `Drop` exclusive access at the
    // last reference, so no lock is needed.
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    _thread: thread::JoinHandle<()>,
}

impl ProviderExecutor {
    /// Start a dedicated provider I/O runtime thread.
    pub fn new() -> CogentResult<Self> {
        Ok(Self {
            inner: Arc::new(start_provider_runtime_thread()?),
        })
    }

    pub(crate) fn spawn<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.inner.handle.spawn(future)
    }
}

impl Drop for ProviderRuntimeThread {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

fn start_provider_runtime_thread() -> CogentResult<ProviderRuntimeThread> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = thread::Builder::new()
        .name("cogentlm-provider-runtime".to_string())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let handle = runtime.handle().clone();
            let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
            if ready_tx.send(Ok((handle, shutdown_tx))).is_err() {
                return;
            }
            let _ = runtime.block_on(shutdown_rx);
        })
        .map_err(|error| {
            CogentError::Internal(format!("failed to spawn provider runtime: {error}"))
        })?;

    let (handle, shutdown_tx) = ready_rx
        .recv()
        .map_err(|_| CogentError::Internal("provider runtime stopped before startup".to_string()))?
        .map_err(|error| {
            CogentError::Internal(format!("failed to build provider runtime: {error}"))
        })?;

    Ok(ProviderRuntimeThread {
        handle,
        shutdown_tx: Some(shutdown_tx),
        _thread: thread,
    })
}
