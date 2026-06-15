mod emscripten;
mod macos;
mod unix;
mod windows;

use crate::build_support::context::BuildContext;
use cmake::Config;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../src/tests/build_support/targets_tests.rs"]
mod targets_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TargetKind {
    Windows,
    Macos,
    Emscripten,
    Unix,
}

impl TargetKind {
    pub(crate) fn is_emscripten(self) -> bool {
        self == Self::Emscripten
    }
}

pub(crate) fn classify(target_os: &str, target: &str) -> TargetKind {
    if target_os == "emscripten" || target.contains("emscripten") {
        TargetKind::Emscripten
    } else if target_os == "windows" {
        TargetKind::Windows
    } else if target_os == "macos" {
        TargetKind::Macos
    } else {
        TargetKind::Unix
    }
}

pub(crate) fn apply_host_cmake_overrides(context: &BuildContext, config: &mut Config) {
    if context.host_is_windows {
        windows::apply_host_cmake_overrides(context, config);
    }
    if context.target_kind == TargetKind::Macos {
        macos::apply_cmake_overrides(context, config);
    }
}

pub(crate) fn apply_cuda_cmake_overrides(context: &BuildContext, config: &mut Config) {
    if context.host_is_windows {
        windows::apply_cuda_cmake_overrides(config);
    }
}

pub(crate) fn link_system_libraries(context: &BuildContext) {
    match context.target_kind {
        TargetKind::Windows => windows::link_system_libraries(context),
        TargetKind::Macos => macos::link_system_libraries(context),
        TargetKind::Emscripten => emscripten::link_system_libraries(),
        TargetKind::Unix => unix::link_system_libraries(context),
    }
}
