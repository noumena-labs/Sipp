use crate::engine::protocol::{EngineEvent, EngineState, EngineStatus, ModelState};
use crate::runtime::InferenceRuntime;

use crate::runtime::numeric::unix_time_ms;

use super::stats::{engine_stats_from_runtime, read_backend_info};
use super::EngineEventSubscribers;

pub(super) fn build_engine_state_with_status(
    runtime: &InferenceRuntime,
    model_state: &ModelState,
    status: Option<EngineStatus>,
) -> EngineState {
    EngineState {
        status: status.unwrap_or_else(|| default_runtime_status(runtime)),
        model: Some(model_state.clone()),
        backend: read_backend_info(),
        runtime: Some(runtime.resolved_runtime_limits()),
        requests: Vec::new(),
        stats: runtime
            .try_get_runtime_observability()
            .map(engine_stats_from_runtime)
            .unwrap_or_default(),
        updated_at_unix_ms: unix_time_ms(),
    }
}

fn default_runtime_status(runtime: &InferenceRuntime) -> EngineStatus {
    if runtime.is_ready() {
        EngineStatus::Ready
    } else {
        EngineStatus::Error
    }
}

pub(super) fn emit_state_event(
    runtime: &InferenceRuntime,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
    status: EngineStatus,
) {
    emit_event(
        event_subscribers,
        EngineEvent::State(Box::new(build_engine_state_with_status(
            runtime,
            model_state,
            Some(status),
        ))),
    );
}

pub(super) fn emit_event(event_subscribers: &EngineEventSubscribers, event: EngineEvent) {
    let Ok(mut subscribers) = event_subscribers.lock() else {
        return;
    };
    subscribers.retain(|subscriber| subscriber.send(event.clone()).is_ok());
}

#[cfg(test)]
mod tests {
    use std::sync::{mpsc, Arc, Mutex};

    use crate::engine::protocol::EngineEvent;

    use super::emit_event;

    #[test]
    fn emit_event_drops_closed_subscribers() {
        let subscribers = Arc::new(Mutex::new(Vec::new()));
        let (closed_tx, closed_rx) = mpsc::channel();
        drop(closed_rx);
        let (open_tx, open_rx) = mpsc::channel();
        subscribers.lock().unwrap().push(closed_tx);
        subscribers.lock().unwrap().push(open_tx);

        emit_event(&subscribers, EngineEvent::Closed);

        assert!(matches!(open_rx.recv().unwrap(), EngineEvent::Closed));
        assert_eq!(subscribers.lock().unwrap().len(), 1);
    }
}
