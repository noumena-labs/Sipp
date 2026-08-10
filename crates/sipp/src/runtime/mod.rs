pub mod config;
pub(crate) mod inference_runtime;
pub mod llama;
pub mod metrics;
pub(crate) mod numeric;
pub mod request;
mod residency;
pub(crate) mod scheduler;
pub(crate) mod session;

pub use inference_runtime::{InferenceRuntime, RequestStepResult, SchedulerBurstResult};
pub use sipp_sys::{llama_seq_id, llama_token};

/// Complete mono PCM16 WAV output returned by speech synthesis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthesizedAudio {
    pub(crate) data: Vec<u8>,
    pub(crate) sample_count: u64,
    pub(crate) sample_rate_hz: u32,
}

impl SynthesizedAudio {
    /// Return the encoded WAV payload.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Return the number of decoded mono samples.
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Return the decoded sample rate in hertz.
    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    /// Return the number of output channels.
    pub fn channels(&self) -> u16 {
        1
    }

    /// Return the decoded audio duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.sample_count * 1_000 / u64::from(self.sample_rate_hz)
    }
}

pub(crate) const REQUEST_CANCELLED_MESSAGE: &str = "Request cancelled.";
