use std::future::Future;
use std::sync::{mpsc, Arc};
use std::thread;

use crate::{CogentError, CogentResult};

/// Owned remote I/O executor used by remote endpoints.
#[derive(Clone)]
pub(crate) struct RemoteExecutor {
    inner: Arc<RemoteRuntimeThread>,
}

struct RemoteRuntimeThread {
    handle: tokio::runtime::Handle,
    // Only `Drop` touches this, and `Arc` gives `Drop` exclusive access at the
    // last reference, so no lock is needed.
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    _thread: thread::JoinHandle<()>,
}

impl RemoteExecutor {
    /// Start a dedicated remote I/O runtime thread.
    pub fn new() -> CogentResult<Self> {
        Ok(Self {
            inner: Arc::new(start_remote_runtime_thread()?),
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

impl Drop for RemoteRuntimeThread {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

fn start_remote_runtime_thread() -> CogentResult<RemoteRuntimeThread> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = thread::Builder::new()
        .name("cogentlm-remote-runtime".to_string())
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
            CogentError::Internal(format!("failed to spawn remote runtime: {error}"))
        })?;

    let (handle, shutdown_tx) = ready_rx
        .recv()
        .map_err(|_| CogentError::Internal("remote runtime stopped before startup".to_string()))?
        .map_err(|error| {
            CogentError::Internal(format!("failed to build remote runtime: {error}"))
        })?;

    Ok(RemoteRuntimeThread {
        handle,
        shutdown_tx: Some(shutdown_tx),
        _thread: thread,
    })
}
