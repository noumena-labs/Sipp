use crate::build_support::context::BuildContext;
use cmake::Config;
use std::env;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../../src/tests/build_support/apple_tests.rs"]
mod apple_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
pub(super) fn apply_cmake_overrides(context: &BuildContext, config: &mut Config) {
    let target = context.target.as_str();
    let deployment_target = deployment_target(target);
    config.define("CMAKE_OSX_ARCHITECTURES", architecture(target));
    config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target);

    if target.ends_with("apple-darwin") {
        env::set_var("MACOSX_DEPLOYMENT_TARGET", &deployment_target);
    } else {
        env::set_var("IPHONEOS_DEPLOYMENT_TARGET", &deployment_target);
        config.define("CMAKE_SYSTEM_NAME", "iOS");
        config.define("CMAKE_OSX_SYSROOT", ios_sdk(target));
    }
}

pub(super) fn architecture(target: &str) -> &'static str {
    match target {
        "aarch64-apple-darwin" | "aarch64-apple-ios" | "aarch64-apple-ios-sim" => "arm64",
        "x86_64-apple-darwin" | "x86_64-apple-ios" => "x86_64",
        _ => panic!("unsupported Apple target: {target}"),
    }
}

fn deployment_target(target: &str) -> String {
    let (variable, default) = if target.ends_with("apple-darwin") {
        ("MACOSX_DEPLOYMENT_TARGET", default_macos_target(target))
    } else {
        ("IPHONEOS_DEPLOYMENT_TARGET", "16.0")
    };
    env::var(variable)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| default.to_owned())
}

fn default_macos_target(target: &str) -> &'static str {
    match target {
        "aarch64-apple-darwin" => "11.0",
        "x86_64-apple-darwin" => "10.15",
        _ => panic!("unsupported macOS target: {target}"),
    }
}

fn ios_sdk(target: &str) -> &'static str {
    match target {
        "aarch64-apple-ios" => "iphoneos",
        "aarch64-apple-ios-sim" | "x86_64-apple-ios" => "iphonesimulator",
        _ => panic!("unsupported iOS target: {target}"),
    }
}

pub(crate) fn link_system_libraries(context: &BuildContext) {
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    if context.features.backend_dl {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    }

    if !context.features.backend_dl && context.features.metal {
        for framework in [
            "Foundation",
            "Metal",
            "MetalKit",
            "QuartzCore",
            "CoreGraphics",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    }

    if !context.features.backend_dl && context.features.vulkan {
        link_vulkan_libraries(context);
    }
}

fn link_vulkan_libraries(context: &BuildContext) {
    if let Some(vulkan_sdk) = &context.env_vars.vulkan_sdk {
        println!(
            "cargo:rustc-link-search=native={}",
            vulkan_sdk.join("lib").display()
        );
    }
    println!("cargo:rustc-link-lib=vulkan");
}
