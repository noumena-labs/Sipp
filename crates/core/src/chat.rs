#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChatRole;

    #[test]
    fn chat_role_serde_uses_snake_case() {
        let encoded = serde_json::to_string(&ChatRole::System).expect("serialize role");
        assert_eq!(encoded, "\"system\"");

        let decoded: ChatRole = serde_json::from_str("\"assistant\"").expect("deserialize role");
        assert_eq!(decoded, ChatRole::Assistant);
    }
}
