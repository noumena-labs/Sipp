// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Sipp",
    platforms: [
        .macOS(.v11),
        .iOS(.v16),
    ],
    products: [
        .library(name: "Sipp", targets: ["Sipp"]),
    ],
    targets: [
        .binaryTarget(
            name: "SippCore",
            path: "Binary/SippCore.xcframework"
        ),
        .target(
            name: "SippCoreBindings",
            dependencies: ["SippCore"],
            path: "Sources/SippCoreBindings"
        ),
        .target(
            name: "Sipp",
            dependencies: ["SippCoreBindings"],
            path: "Sources/Sipp"
        ),
        .testTarget(
            name: "SippTests",
            dependencies: ["Sipp", "SippCoreBindings"],
            path: "Tests/SippTests"
        ),
    ]
)
