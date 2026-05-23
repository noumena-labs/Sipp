//! Unit tests for the parent module.

use super::super::*;

fn fake_template(messages_json: &str, add_assistant: bool) -> Result<String> {
    let messages: Vec<serde_json::Value> =
        serde_json::from_str(messages_json).expect("messages json");
    let mut out = String::new();
    for message in messages {
        let role = message["role"].as_str().expect("role");
        let content = message["content"].as_str().expect("content");
        out.push_str(&format!("<{role}>\n{content}</{role}>\n"));
    }
    if add_assistant {
        out.push_str("<assistant>\n");
    }
    Ok(out)
}

#[test]
fn probes_boundaries_without_system_only_template_call() {
    let mut observed_roles = Vec::new();
    let info = probe_chat_boundary_info(
        |messages_json, add_assistant| {
            let messages: Vec<serde_json::Value> =
                serde_json::from_str(messages_json).expect("messages json");
            let roles: Vec<_> = messages
                .iter()
                .map(|message| message["role"].as_str().expect("role").to_string())
                .collect();
            assert_ne!(roles, vec!["system".to_string()]);
            observed_roles.push((roles, add_assistant));
            fake_template(messages_json, add_assistant)
        },
        "</s>",
    )
    .expect("boundary info");

    assert_eq!(info.assistant_prefix, "<assistant>\n");
    assert_eq!(info.assistant_suffix, "</assistant>\n");
    assert_eq!(
        info.next_turn_prefixes,
        vec!["<system>\n", "<user>\n", "<assistant>\n"]
    );
    assert_eq!(info.eos_text, "</s>");
    assert_eq!(observed_roles.len(), 4);
}
