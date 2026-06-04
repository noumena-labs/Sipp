use crate::engine::protocol::{EmbeddingCapabilities, ModelCapabilities, ModelClass, PoolingType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeModelCapabilities {
    pub class: ModelClass,
    pub embedding_dimensions: i32,
    pub pooling_type: PoolingType,
    pub decoder_start_token: Option<i32>,
    pub has_chat_template: bool,
    pub embedding_context: bool,
}

impl RuntimeModelCapabilities {
    pub(crate) fn supports_text_generation(&self) -> bool {
        match self.class {
            ModelClass::DecoderOnly => !self.embedding_context,
            ModelClass::EncoderDecoder => true,
            ModelClass::EncoderOnly => false,
        }
    }

    pub(crate) fn supports_embeddings(&self) -> bool {
        if self.pooling_type == PoolingType::None {
            return false;
        }
        match self.class {
            ModelClass::DecoderOnly => self.embedding_context,
            ModelClass::EncoderDecoder => false,
            ModelClass::EncoderOnly => true,
        }
    }

    pub(crate) fn to_public(&self) -> ModelCapabilities {
        ModelCapabilities {
            model_class: self.class,
            supports_text_generation: self.supports_text_generation(),
            supports_embeddings: self.supports_embeddings(),
            has_chat_template: self.has_chat_template,
            embedding: self.supports_embeddings().then_some(EmbeddingCapabilities {
                dimensions: self.embedding_dimensions,
                pooling: self.pooling_type,
            }),
        }
    }
}

#[cfg(test)]
#[path = "../../tests/runtime/inference_runtime/capabilities_tests.rs"]
mod capabilities_tests;
