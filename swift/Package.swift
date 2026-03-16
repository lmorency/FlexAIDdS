// swift-tools-version: 5.9
// Package.swift — FlexAIDdS Swift Package
//
// Native Swift wrapper for the FlexAID entropy-driven molecular docking engine.
// Provides: statistical mechanics, vibrational entropy, fleet scheduling,
// HealthKit, MusicKit, Apple Intelligence integrations.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import PackageDescription

let package = Package(
    name: "FlexAIDdS",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "FlexAIDdS", targets: ["FlexAIDdS"]),
        .library(name: "FleetScheduler", targets: ["FleetScheduler"]),
        .library(name: "HealthIntegration", targets: ["HealthIntegration"]),
        .library(name: "MediaIntegration", targets: ["MediaIntegration"]),
        .library(name: "Intelligence", targets: ["Intelligence"]),
    ],
    targets: [
        // Layer 1: C/Obj-C++ bridge to the C++ core
        // NOTE: For full builds, the C++ sources from LIB/ must be copied here
        // or a pre-built libFlexAIDCore.a must be linked via CMake.
        // This target exposes only the C headers to Swift.
        .target(
            name: "FlexAIDCore",
            path: "Sources/FlexAIDCore",
            publicHeadersPath: "include",
            cxxSettings: [
                .headerSearchPath("../../LIB"),
                .define("FLEXAIDS_SWIFT_BRIDGE"),
                .unsafeFlags(["-std=c++20"]),
            ],
            linkerSettings: [
                .linkedLibrary("c++"),
            ]
        ),

        // Layer 2: Swift module — actors, models, Swift-native API
        .target(
            name: "FlexAIDdS",
            dependencies: ["FlexAIDCore"],
            path: "Sources/FlexAIDdS"
        ),

        // Layer 3: Feature modules
        .target(
            name: "FleetScheduler",
            dependencies: ["FlexAIDdS", "Intelligence"],
            path: "Sources/FleetScheduler"
        ),
        .target(
            name: "HealthIntegration",
            dependencies: ["FlexAIDdS"],
            path: "Sources/HealthIntegration"
        ),
        .target(
            name: "MediaIntegration",
            dependencies: ["FlexAIDdS", "HealthIntegration"],
            path: "Sources/MediaIntegration"
        ),
        .target(
            name: "Intelligence",
            dependencies: ["FlexAIDdS", "HealthIntegration"],
            path: "Sources/Intelligence"
        ),

        // Tests
        .testTarget(
            name: "FlexAIDdSTests",
            dependencies: ["FlexAIDdS", "Intelligence"],
            path: "Tests/FlexAIDdSTests"
        ),
    ]
)
