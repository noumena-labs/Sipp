/// Tri-state capability used by providers and endpoint resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilitySupport {
    /// The endpoint is known to support the operation.
    Supported,
    /// The endpoint is known not to support the operation.
    Unsupported,
    /// The endpoint's support is not known without attempting the operation.
    Unknown,
}
