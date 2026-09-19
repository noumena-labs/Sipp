//! Markdown code-fence import validation for public package documentation.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, Result};
use regex::Regex;

use crate::utils::BuildContext;

use super::collect_files_with_extension;

const INTERNAL_TYPESCRIPT_PACKAGES: &[&str] = &["@noumena-labs/sipp", "@noumena-labs/sipp-server"];

pub(super) struct PublicSymbols {
    pub(super) web: BTreeSet<String>,
    pub(super) web_vite: BTreeSet<String>,
    pub(super) node: BTreeSet<String>,
    pub(super) python: BTreeSet<String>,
}

pub(super) struct DocSnippetChecker {
    symbols: PublicSymbols,
    patterns: SnippetPatterns,
}

impl DocSnippetChecker {
    pub(super) fn new(symbols: PublicSymbols) -> Result<Self> {
        Ok(Self {
            symbols,
            patterns: SnippetPatterns::new()?,
        })
    }

    pub(super) fn check(&self, markdown: &str, display: &str, violations: &mut Vec<String>) {
        check_doc_snippet_imports_with_patterns(
            markdown,
            display,
            &self.symbols,
            &self.patterns,
            violations,
        );
    }
}

struct SnippetPatterns {
    fence: Regex,
    typescript_import: Regex,
    python_import: Regex,
}

impl SnippetPatterns {
    fn new() -> Result<Self> {
        Ok(Self {
            fence: compile_regex(
                r"(?ms)^```(?P<info>[^\r\n]*)\r?\n(?P<body>.*?)^```[ \t]*\r?$",
                "Markdown code fence",
            )?,
            typescript_import: compile_regex(
                r#"(?ms)^[ \t]*import\s*\{(?P<items>[^}]*)\}\s*from\s*[\"'](?P<package>[^\"']+)[\"']"#,
                "TypeScript import",
            )?,
            python_import: compile_regex(
                r"(?ms)^\s*from\s+sipp\s+import\s+(?:\((?P<group>[^)]*)\)|(?P<line>[^\r\n]+))",
                "Python import",
            )?,
        })
    }
}

pub(super) fn collect_docs_code_fence_symbol_violations(
    ctx: &BuildContext,
    violations: &mut Vec<String>,
) -> Result<()> {
    let docs_dir = ctx.workspace_root().join("docs");
    if !docs_dir.is_dir() {
        anyhow::bail!(
            "public documentation directory is missing: {}",
            docs_dir.display()
        );
    }

    let checker = DocSnippetChecker::new(load_public_symbols(ctx)?)?;
    for path in collect_files_with_extension(&docs_dir, "md")? {
        let display = path
            .strip_prefix(ctx.workspace_root())
            .unwrap_or(&path)
            .display()
            .to_string();
        let markdown = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        checker.check(&markdown, &display, violations);
    }
    Ok(())
}

fn check_doc_snippet_imports_with_patterns(
    markdown: &str,
    display: &str,
    symbols: &PublicSymbols,
    patterns: &SnippetPatterns,
    violations: &mut Vec<String>,
) {
    for captures in patterns.fence.captures_iter(markdown) {
        let Some(info) = captures.name("info") else {
            continue;
        };
        let Some(body) = captures.name("body") else {
            continue;
        };
        let language = info.as_str().split_whitespace().next().unwrap_or("");
        let body_line = line_number(markdown, body.start());
        match language {
            "ts" | "typescript" | "js" | "javascript" => check_typescript_imports(
                body.as_str(),
                display,
                body_line,
                symbols,
                &patterns.typescript_import,
                violations,
            ),
            "python" | "py" => check_python_imports(
                body.as_str(),
                display,
                body_line,
                &symbols.python,
                &patterns.python_import,
                violations,
            ),
            _ => {}
        }
    }
}

fn check_typescript_imports(
    code: &str,
    display: &str,
    body_line: usize,
    symbols: &PublicSymbols,
    pattern: &Regex,
    violations: &mut Vec<String>,
) {
    for captures in pattern.captures_iter(code) {
        let Some(import_match) = captures.get(0) else {
            continue;
        };
        let Some(items) = captures.name("items") else {
            continue;
        };
        let Some(package) = captures.name("package") else {
            continue;
        };
        let package = package.as_str();
        let import_line = body_line + newline_count(&code[..import_match.start()]);
        if package_matches_any(package, INTERNAL_TYPESCRIPT_PACKAGES) {
            violations.push(format!(
                "{display}:{import_line}: public snippet imports inaccessible internal package `{package}`"
            ));
            continue;
        }

        let expected = match package {
            "@sipphq/sipp" => Some(&symbols.web),
            "@sipphq/sipp/vite" => Some(&symbols.web_vite),
            "@sipphq/sipp-server" => Some(&symbols.node),
            _ => None,
        };
        let Some(expected) = expected else {
            if package_matches_any(package, &["@sipphq/sipp", "@sipphq/sipp-server"]) {
                violations.push(format!(
                    "{display}:{import_line}: public snippet imports unsupported Sipp package entry point `{package}`"
                ));
            }
            continue;
        };
        for name in imported_typescript_names(items.as_str()) {
            if !expected.contains(name) {
                violations.push(format!(
                    "{display}:{import_line}: snippet imports `{name}` from `{package}`, but `{name}` is not in the exported public API"
                ));
            }
        }
    }
}

fn check_python_imports(
    code: &str,
    display: &str,
    body_line: usize,
    expected: &BTreeSet<String>,
    pattern: &Regex,
    violations: &mut Vec<String>,
) {
    for captures in pattern.captures_iter(code) {
        let Some(import_match) = captures.get(0) else {
            continue;
        };
        let items = captures
            .name("group")
            .or_else(|| captures.name("line"))
            .map(|value| value.as_str())
            .unwrap_or("");
        let import_line = body_line + newline_count(&code[..import_match.start()]);
        for item in items.split(',') {
            let Some(name) = item.split_whitespace().next() else {
                continue;
            };
            let name = trim_identifier(name);
            if !name.is_empty() && !expected.contains(name) {
                violations.push(format!(
                    "{display}:{import_line}: snippet imports `{name}` from `sipp`, but `{name}` is not in the exported public API"
                ));
            }
        }
    }
}

fn load_public_symbols(ctx: &BuildContext) -> Result<PublicSymbols> {
    let root = ctx.workspace_root();
    let web = parse_typescript_exports(&root.join("lib/web/src/index.ts"))?;
    let web_vite = parse_typescript_exports(&root.join("lib/web/src/vite.ts"))?;
    let mut node = parse_typescript_exports(&root.join("lib/node/index.d.ts"))?;
    node.extend(parse_typescript_exports(
        &root.join("lib/node/router.d.ts"),
    )?);
    let python = parse_python_exports(&root.join("lib/python/python/sipp/__init__.py"))?;
    Ok(PublicSymbols {
        web,
        web_vite,
        node,
        python,
    })
}

fn parse_typescript_exports(path: &Path) -> Result<BTreeSet<String>> {
    let source = read_source(path)?;
    let export_list = compile_regex(
        r"(?ms)^[ \t]*export\s+(?:type\s+)?\{(?P<items>[^}]*)\}",
        "TypeScript export list",
    )?;
    let declaration = compile_regex(
        r"(?m)^\s*export\s+(?:declare\s+)?(?:class|interface|type|const|function|enum)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)",
        "TypeScript export declaration",
    )?;
    let mut exports = BTreeSet::new();

    for captures in export_list.captures_iter(&source) {
        let Some(items) = captures.name("items") else {
            continue;
        };
        for item in items.as_str().split(',') {
            let tokens = item.split_whitespace().collect::<Vec<_>>();
            let name = match tokens.as_slice() {
                ["type", _, "as", alias, ..] | [_, "as", alias, ..] => *alias,
                ["type", name, ..] | [name, ..] => *name,
                [] => continue,
            };
            let name = trim_identifier(name);
            if !name.is_empty() {
                exports.insert(name.to_string());
            }
        }
    }
    for captures in declaration.captures_iter(&source) {
        if let Some(name) = captures.name("name") {
            exports.insert(name.as_str().to_string());
        }
    }
    Ok(exports)
}

fn parse_python_exports(path: &Path) -> Result<BTreeSet<String>> {
    let source = read_source(path)?;
    let all_list = compile_regex(
        r"(?s)__all__\s*=\s*\[(?P<items>.*?)\]",
        "Python __all__ list",
    )?;
    let quoted_name = compile_regex(
        r#"[\"'](?P<name>[A-Za-z_][A-Za-z0-9_]*)[\"']"#,
        "Python exported name",
    )?;
    let declaration = compile_regex(
        r"(?m)^(?:class|def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
        "Python public declaration",
    )?;
    let mut exports = BTreeSet::new();

    if let Some(captures) = all_list.captures(&source) {
        if let Some(items) = captures.name("items") {
            for captures in quoted_name.captures_iter(items.as_str()) {
                if let Some(name) = captures.name("name") {
                    exports.insert(name.as_str().to_string());
                }
            }
        }
    }
    for captures in declaration.captures_iter(&source) {
        if let Some(name) = captures.name("name") {
            if !name.as_str().starts_with('_') {
                exports.insert(name.as_str().to_string());
            }
        }
    }
    Ok(exports)
}

fn imported_typescript_names(items: &str) -> impl Iterator<Item = &str> {
    items.split(',').filter_map(|item| {
        let mut tokens = item.split_whitespace();
        let first = tokens.next()?;
        let name = if first == "type" {
            tokens.next().unwrap_or("")
        } else {
            first
        };
        let name = trim_identifier(name);
        (!name.is_empty()).then_some(name)
    })
}

fn read_source(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

fn package_matches_any(package: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| {
        package == *candidate
            || package
                .strip_prefix(candidate)
                .is_some_and(|suffix| suffix.starts_with('/'))
    })
}

fn compile_regex(pattern: &str, label: &str) -> Result<Regex> {
    Regex::new(pattern).with_context(|| format!("invalid built-in {label} regex"))
}

fn trim_identifier(value: &str) -> &str {
    value.trim_matches(|character: char| !character.is_alphanumeric() && character != '_')
}

fn line_number(source: &str, byte_offset: usize) -> usize {
    newline_count(&source[..byte_offset]) + 1
}

fn newline_count(value: &str) -> usize {
    value.bytes().filter(|byte| *byte == b'\n').count()
}
