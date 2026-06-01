use std::path::PathBuf;

pub(crate) fn sanitize_path(path: impl AsRef<str>) -> PathBuf {
    let path = path.as_ref();
    if let Some(stripped) = path.strip_prefix(r"\\?\") {
        PathBuf::from(stripped)
    } else {
        PathBuf::from(path)
    }
}

pub(crate) fn path_component(value: &str, fallback: &str) -> String {
    let source = if value.is_empty() { fallback } else { value };
    source
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' => '_',
            _ => ch,
        })
        .collect()
}
