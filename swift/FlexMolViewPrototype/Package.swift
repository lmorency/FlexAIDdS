// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FlexMolViewPrototype",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "FlexMolViewPrototype", targets: ["FlexMolViewPrototype"]),
    ],
    targets: [
        .target(
            name: "FlexMolViewPrototype",
            path: "Sources"
        ),
        .testTarget(
            name: "FlexMolViewPrototypeTests",
            dependencies: ["FlexMolViewPrototype"],
            path: "Tests"
        ),
    ]
)
