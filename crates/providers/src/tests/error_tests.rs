//! Unit tests for the parent module.

use super::*;

#[test]
fn provider_error_kind_from_code_maps_retry_relevant_codes() {
    assert_eq!(
        provider_error_kind_from_code(Some("rate_limit")),
        Some(ProviderErrorKind::RateLimited)
    );
    assert_eq!(
        provider_error_kind_from_code(Some("insufficient_quota")),
        Some(ProviderErrorKind::QuotaExceeded)
    );
    assert_eq!(
        provider_error_kind_from_code(Some("overloaded_error")),
        Some(ProviderErrorKind::Overloaded)
    );
    assert_eq!(provider_error_kind_from_code(Some("bad_request")), None);
}
