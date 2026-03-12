// DeviceCapability.swift — Device capability detection for fleet scheduling
//
// Detects hardware capabilities, thermal state, and compute capacity.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Hardware capabilities of a device in the Bonhomme fleet.
public struct DeviceCapability: Sendable, Codable, Hashable {
    /// Unique device identifier (hashed for privacy)
    public let deviceID: String

    /// Device model identifier (e.g., "MacBookPro18,3")
    public let model: String

    /// Estimated TFLOPS for the device GPU
    public let estimatedTFLOPS: Double

    /// Whether the device has a discrete/integrated GPU
    public let hasGPU: Bool

    /// Available memory in GB
    public let availableMemoryGB: Double

    /// Current thermal state
    public let thermalState: ThermalState

    /// Compute share weight (0.0-1.0) based on capability
    public let computeWeight: Double

    /// Thermal state mirroring ProcessInfo.ThermalState for Codable support.
    public enum ThermalState: String, Sendable, Codable, Hashable {
        case nominal
        case fair
        case serious
        case critical
    }

    /// Detect capabilities of the current device.
    public static func current() -> DeviceCapability {
        let processInfo = ProcessInfo.processInfo

        let thermalState: ThermalState = {
            switch processInfo.thermalState {
            case .nominal: return .nominal
            case .fair: return .fair
            case .serious: return .serious
            case .critical: return .critical
            @unknown default: return .nominal
            }
        }()

        let memoryGB = Double(processInfo.physicalMemory) / (1024 * 1024 * 1024)

        // Estimate TFLOPS based on available processors
        let cores = processInfo.processorCount
        let estimatedTFLOPS = Double(cores) * 0.1  // Conservative estimate

        // Compute weight based on thermal state and capability
        let thermalMultiplier: Double = {
            switch thermalState {
            case .nominal: return 1.0
            case .fair: return 0.75
            case .serious: return 0.4
            case .critical: return 0.0
            }
        }()

        let weight = min(1.0, estimatedTFLOPS / 10.0) * thermalMultiplier

        return DeviceCapability(
            deviceID: Self.hashedDeviceID(),
            model: Self.modelIdentifier(),
            estimatedTFLOPS: estimatedTFLOPS,
            hasGPU: true,  // All Apple Silicon has GPU
            availableMemoryGB: memoryGB,
            thermalState: thermalState,
            computeWeight: weight
        )
    }

    private static func hashedDeviceID() -> String {
        let hostName = ProcessInfo.processInfo.hostName
        return String(hostName.hashValue, radix: 16)
    }

    private static func modelIdentifier() -> String {
        #if os(macOS)
        var size: size_t = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var model = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.model", &model, &size, nil, 0)
        return String(cString: model)
        #else
        return ProcessInfo.processInfo.hostName
        #endif
    }
}
