use crate::engine::protocol::{EmbeddingCapabilities, ModelCapabilities, ModelClass, PoolingType};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeModelCapabilities {
    pub class: ModelClass,
    pub n_embd: i32,
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
                dimensions: self.n_embd,
                pooling: self.pooling_type,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_capabilities_hide_decoder_start_token() {
        let capabilities = RuntimeModelCapabilities {
            class: ModelClass::EncoderDecoder,
            n_embd: 768,
            pooling_type: PoolingType::Mean,
            decoder_start_token: Some(0),
            has_chat_template: false,
            embedding_context: false,
        }
        .to_public();

        assert_eq!(capabilities.model_class, ModelClass::EncoderDecoder);
        assert!(capabilities.supports_text_generation);
        assert!(!capabilities.supports_embeddings);
        assert!(capabilities.embedding.is_none());
    }

    #[test]
    fn public_capabilities_include_embedding_metadata_only_when_supported() {
        let capabilities = RuntimeModelCapabilities {
            class: ModelClass::EncoderOnly,
            n_embd: 1024,
            pooling_type: PoolingType::Cls,
            decoder_start_token: None,
            has_chat_template: false,
            embedding_context: true,
        }
        .to_public();

        assert!(!capabilities.supports_text_generation);
        assert!(capabilities.supports_embeddings);
        assert_eq!(
            capabilities.embedding,
            Some(EmbeddingCapabilities {
                dimensions: 1024,
                pooling: PoolingType::Cls,
            })
        );
    }
}
