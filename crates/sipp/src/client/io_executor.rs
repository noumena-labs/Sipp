use std::future::Future;
use std::sync::{mpsc, Arc};
use std::thread;

use crate::client::{SippError, SippResult};

/// Owned asynchronous I/O executor used by built-in gateway endpoints.
#[derive(Clone)]
pub(crate) struct IoExecutor {
    inner: Arc<IoRuntimeThread>,
}

struct IoRuntimeThread {
    handle: tokio::runtime::Handle,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    _thread: thread::JoinHandle<()>,
}

impl IoExecutor {
    pub(crate) fn new() -> SippResult<Self> {
        Ok(Self {
            inner: Arc::new(start_io_runtime_thread()?),
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

impl Drop for IoRuntimeThread {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
    }
}

fn start_io_runtime_thread() -> SippResult<IoRuntimeThread> {
    let (ready_tx, ready_rx) = mpsc::sync_channel(1);
    let thread = thread::Builder::new()
        .name("sipp-io-runtime".to_string())
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
        .map_err(|error| SippError::Internal(format!("failed to spawn I/O runtime: {error}")))?;

    let (handle, shutdown_tx) = ready_rx
        .recv()
        .map_err(|_| SippError::Internal("I/O runtime stopped before startup".to_string()))?
        .map_err(|error| SippError::Internal(format!("failed to build I/O runtime: {error}")))?;

    Ok(IoRuntimeThread {
        handle,
        shutdown_tx: Some(shutdown_tx),
        _thread: thread,
    })
}
