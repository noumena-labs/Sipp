use std::mem::size_of;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrowserSchedulerLoopResult {
    pub ticks_executed: i32,
    pub progressed_ticks: i32,
    pub completed_response_count: i32,
    pub emitted_token_count: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct BrowserRuntimeMetrics {
    pub ttft_ms: f64,
    pub itl_avg_ms: f64,
    pub itl_p99_ms: f64,
    pub e2e_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub native_gpu_ms: f64,
    pub native_sync_ms: f64,
    pub native_logic_ms: f64,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
}

const _: () = assert!(size_of::<BrowserSchedulerLoopResult>() == 16);
const _: () = assert!(size_of::<BrowserRuntimeMetrics>() == 88);
