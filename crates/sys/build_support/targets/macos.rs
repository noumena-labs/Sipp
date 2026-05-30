use crate::build_support::context::BuildContext;

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
