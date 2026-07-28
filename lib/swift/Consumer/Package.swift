// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "SippConsumer",
    platforms: [
        .macOS(.v11),
    ],
    dependencies: [
        .package(name: "Sipp", path: "../package"),
    ],
    targets: [
        .executableTarget(
            name: "SippConsumer",
            dependencies: [
                .product(name: "Sipp", package: "Sipp"),
            ]
        ),
    ]
)
