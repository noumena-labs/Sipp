use crate::build_support::context::BuildContext;
use cmake::Config;
use std::env;

pub(super) fn apply_cmake_overrides(context: &BuildContext, config: &mut Config) {
    let deployment_target = deployment_target(context);
    env::set_var("MACOSX_DEPLOYMENT_TARGET", deployment_target);
    config.define("CMAKE_OSX_DEPLOYMENT_TARGET", deployment_target);
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
        let lib_dir = vulkan_sdk.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=vulkan");
}

pub(super) fn deployment_target(context: &BuildContext) -> &'static str {
    if context.target.contains("aarch64") || context.target.contains("arm64") {
        "11.0"
    } else {
        "10.15"
    }
}
