//! Unit tests for the parent module.

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
