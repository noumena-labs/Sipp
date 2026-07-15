//! Shared remote-model acquisition protocol and policy.

mod policy;
mod state;
mod types;

#[cfg(not(target_family = "wasm"))]
pub(crate) mod native;

pub(crate) use state::{RemoteAcquisition, RemoteAcquisitionIds};
pub(crate) use types::{
    canonical_remote_url, RemoteAcquisitionEvent, RemoteAcquisitionProgress,
    RemoteAcquisitionRequest, RemoteAction, RemoteAssetRole, RemoteCacheCandidate, RemoteFailure,
    RemoteFailureKind, RemoteFailurePhase, RemoteMetadata, RemoteMetadataHeaders,
    RemoteResolvedMember,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/acquisition_tests.rs"]
mod acquisition_tests;
