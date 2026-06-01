use assert_cmd::Command;

#[test]
fn help_exposes_user_facing_flags() {
    let output = Command::cargo_bin("cogentlm")
        .expect("cogentlm binary")
        .arg("--help")
        .output()
        .expect("run help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("Usage:"));
    assert!(stdout.contains("--max-tokens"));
    assert!(stdout.contains("--backend"));
    assert!(stdout.contains("--stats"));
}

#[test]
fn missing_required_arguments_fail_before_model_loading() {
    let output = Command::cargo_bin("cogentlm")
        .expect("cogentlm binary")
        .output()
        .expect("run without args");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("utf8 stderr");
    assert!(stderr.contains("required"));
}
