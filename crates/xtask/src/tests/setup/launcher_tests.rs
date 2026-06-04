//! Tests the `setup::launcher` module in `xtask`.
//!
//! Covers launcher quoting, environment script generation, and path comparison
//! helpers with fake paths instead of installing or executing launcher scripts.

use std::path::Path;

use crate::test_support::TempDir;

use super::{
    cmd_env_script, cmd_quote, path_eq, path_string_eq, powershell_env_script, powershell_quote,
    powershell_quote_str, shell_quote, shell_quote_str, unix_env_script,
};

#[test]
fn shell_quote_escapes_single_quotes_and_normalizes_windows_drive_paths() {
    assert_eq!(shell_quote_str("a'b"), "'a'\"'\"'b'");
    let quoted = shell_quote(Path::new("D:\\Cogent LM\\bin"));
    assert_eq!(quoted, "'/d/Cogent LM/bin'");
}

#[test]
fn powershell_and_cmd_quotes_escape_shell_specific_characters() {
    assert_eq!(powershell_quote_str("a'b"), "'a''b'");
    assert_eq!(powershell_quote(Path::new("C:\\bin")), "'C:\\bin'");
    assert_eq!(
        cmd_quote(Path::new("C:\\Program Files\\bin")),
        "\"C:\\Program Files\\bin\""
    );
}

#[test]
fn env_scripts_prepend_launcher_directory_once() {
    let bin = Path::new("D:\\CogentLM\\.build\\bin");
    let unix = unix_env_script(bin);
    assert!(unix.contains("COGENTLM_BIN="));
    assert!(unix.contains("export PATH="));
    assert!(!unix.contains("\r\n"));

    let powershell = powershell_env_script(bin);
    assert!(powershell.contains("$CogentLmBin"));
    assert!(powershell.contains("[System.IO.Path]::PathSeparator"));

    let cmd = cmd_env_script(Path::new("C:\\100%\\bin"));
    assert!(cmd.contains("C:\\100%%\\bin"));
    assert!(cmd.contains("PATH=%COGENTLM_BIN%;%PATH%"));
}

#[test]
fn path_equality_uses_canonical_paths_when_available() {
    let temp = TempDir::new("launcher-path-eq");
    let left = temp.create_dir("bin");
    let right = temp.join(".").join("bin");

    assert!(path_eq(&left, &right));
    assert!(path_string_eq(Path::new("same"), Path::new("same")));
    assert!(!path_string_eq(Path::new("left"), Path::new("right")));
}
