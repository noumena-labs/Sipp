use crate::build_support::context::BuildContext;
use std::io::Write;
use std::path::Path;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../src/tests/build_support/ide_tests.rs"]
mod ide_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) fn setup_clangd_autocomplete(context: &BuildContext, dst: &Path) {
    let comp_db = dst.join("build/compile_commands.json");
    let comp_flags = dst.join("build/compile_flags.txt");
    let ide_dir = context.workspace_build_dir().join("ide").join("sys");
    let compile_commands_path = ide_dir.join("compile_commands.json");
    let compile_flags_path = ide_dir.join("compile_flags.txt");

    if comp_db.exists() {
        let _ = std::fs::create_dir_all(&ide_dir);
        let _ = std::fs::copy(&comp_db, &compile_commands_path);
        let _ = std::fs::remove_file(&compile_flags_path);
        return;
    }

    if !comp_flags.exists() {
        let _ = std::fs::create_dir_all(&ide_dir);
        if let Ok(mut file) = std::fs::File::create(&compile_flags_path) {
            let _ = writeln!(file, "-xc++");
            let _ = writeln!(file, "-std=c++17");
            let _ = writeln!(
                file,
                "-I{}",
                context.manifest_dir.join("native/cxx_bridge").display()
            );
            let _ = writeln!(
                file,
                "-I{}",
                context.manifest_dir.join("native/llama_shim").display()
            );
            let _ = writeln!(file, "-I{}", context.llama_dir.join("include").display());
            let _ = writeln!(
                file,
                "-I{}",
                context.llama_dir.join("ggml/include").display()
            );
            let _ = writeln!(file, "-I{}", context.llama_dir.join("common").display());
            let _ = writeln!(file, "-I{}", context.llama_dir.join("tools/mtmd").display());
            let _ = writeln!(file, "-I{}", context.llama_dir.join("vendor").display());
            let _ = writeln!(file, "-DGGML_USE_OPENMP=OFF");
        }
        let _ = std::fs::remove_file(&compile_commands_path);
    }
}
