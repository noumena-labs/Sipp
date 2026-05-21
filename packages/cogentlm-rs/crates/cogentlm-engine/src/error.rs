use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("string contains an interior NUL byte")]
    InteriorNul(#[from] std::ffi::NulError),

    #[error("failed to load model from {path}")]
    ModelLoad { path: String },

    #[error("failed to create llama context")]
    ContextInit,

    #[error("llama returned null for {0}")]
    NullPointer(&'static str),

    #[error("failed to tokenize text")]
    Tokenize,

    #[error("failed to convert token {token} to text")]
    TokenToPiece { token: i32 },

    #[error("decode failed with status {0}")]
    Decode(i32),

    #[error("batch capacity exceeded: capacity={capacity}, requested={requested}")]
    BatchCapacity { capacity: i32, requested: i32 },

    #[error("prompt has {prompt_tokens} tokens but context allows {context_tokens}")]
    PromptTooLong {
        prompt_tokens: usize,
        context_tokens: u32,
    },

    #[error("sampler initialization failed")]
    SamplerInit,

    #[error("runtime is not ready")]
    RuntimeNotReady,

    #[error("invalid request: {0}")]
    InvalidRequest(&'static str),

    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),

    #[error("runtime command failed: {0}")]
    RuntimeCommand(String),
}

pub type Result<T> = std::result::Result<T, Error>;
