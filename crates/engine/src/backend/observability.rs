use crate::error::Result;

use super::ensure_backend_initialized;

pub fn backend_observability_json(include_details: bool) -> Result<String> {
    ensure_backend_initialized();
    Ok(crate::native_bridge::backend_observability_json(
        include_details,
    ))
}
