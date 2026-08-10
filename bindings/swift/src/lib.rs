#![allow(
    clippy::empty_line_after_doc_comments,
    reason = "test and source section banners follow repository style"
)]
//! UniFFI bridge for the Sipp Swift package.
//!
//! This crate is an internal ABI projection. The public Apple API lives in
//! `lib/swift` and must not expose generated UniFFI declarations directly.

mod bridge;
mod inference;

uniffi::setup_scaffolding!();
