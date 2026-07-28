//! Tests the `toolchains::python` module in `xtask`.
//!
//! Covers managed Python resolution with a fake uv executable so the test does
//! not depend on a host interpreter, user cache, or network access.

#[cfg(unix)]
use crate::test_support::TempDir;

#[cfg(unix)]
use super::{ensure_python, PYTHON_BUILD_VERSION};

#[cfg(unix)]
#[test]
fn ensure_python_returns_the_managed_interpreter() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new("managed-python");
    let ctx = crate::utils::BuildContext::from_workspace_root_for_test(temp.path());
    let python_exe = temp.write(".build/toolchain/python/cpython-3.10/bin/python3", "");
    let uv_exe = temp.write(
        ".build/toolchain/uv/uv",
        format!(
            "#!/bin/sh\n\
             if [ \"$1\" = \"python\" ] && [ \"$2\" = \"install\" ]; then\n\
               [ \"$3\" = \"{PYTHON_BUILD_VERSION}\" ] || exit 1\n\
               exit 0\n\
             fi\n\
             if [ \"$1\" = \"python\" ] && [ \"$2\" = \"find\" ]; then\n\
               [ \"$3\" = \"--managed-python\" ] || exit 1\n\
               [ \"$4\" = \"--no-project\" ] || exit 1\n\
               [ \"$5\" = \"{PYTHON_BUILD_VERSION}\" ] || exit 1\n\
               printf '%s\\n' '{}'\n\
               exit 0\n\
             fi\n\
             exit 1\n",
            python_exe.display()
        ),
    );
    let mut permissions = std::fs::metadata(&uv_exe).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&uv_exe, permissions).unwrap();
    let sh = xshell::Shell::new().unwrap();

    assert_eq!(ensure_python(&sh, &ctx, &uv_exe).unwrap(), python_exe);
}
