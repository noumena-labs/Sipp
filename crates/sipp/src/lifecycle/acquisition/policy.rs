use super::types::retry_after_ms;
use super::{RemoteFailure, RemoteFailureKind};

pub(super) const MAX_ATTEMPTS: u8 = 4;

const RETRY_DELAYS_MS: [u64; 3] = [250, 500, 1_000];

pub(super) fn retry_delay_ms(failure: &RemoteFailure, attempt: u8) -> Option<u64> {
    if attempt >= MAX_ATTEMPTS || !retryable(failure) {
        return None;
    }
    let index = usize::from(attempt - 1);
    let configured = RETRY_DELAYS_MS[index];
    Some(
        retry_after_ms(failure.retry_after.as_deref())
            .unwrap_or(configured)
            .max(configured),
    )
}

fn retryable(failure: &RemoteFailure) -> bool {
    match failure.kind {
        RemoteFailureKind::Transport => true,
        RemoteFailureKind::Http => failure
            .status
            .is_some_and(|status| matches!(status, 408 | 425 | 429 | 500 | 502 | 503 | 504)),
        RemoteFailureKind::InvalidResponse
        | RemoteFailureKind::Integrity
        | RemoteFailureKind::Storage => false,
    }
}
