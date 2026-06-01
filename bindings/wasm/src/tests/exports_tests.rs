//! Unit tests for the parent module.

use super::*;

#[test]
fn mutable_engine_operation_does_not_hold_refcell_borrow() {
    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = Some(Box::new(BrowserEngine::create()));
    });

    let can_borrow_inside_operation = with_current_engine_mut(false, |_| {
        CURRENT_ENGINE.with(|current| current.try_borrow_mut().is_ok())
    });

    assert!(can_borrow_inside_operation);
    assert!(current_engine_initialized());
    close_current_engine();
}
