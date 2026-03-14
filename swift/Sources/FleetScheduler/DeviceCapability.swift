// DeviceCapability.swift — Device capability detection for fleet scheduling
//
// Detects hardware capabilities, thermal state, battery level, and compute capacity.
// Battery-aware scheduling prevents work allocation to low-power unplugged devices.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
#if canImport(IOKit)
import IOKit.ps
#endif
#if canImport(UIKit)
import UIKit
#endif

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

    /// Battery level (0.0 - 1.0), nil for desktop Macs
    public let batteryLevel: Double?

    /// Whether the device is connected to power
    public let isCharging: Bool

    /// Compute share weight (0.0-1.0) based on capability, thermal, and battery
    public let computeWeight: Double

    /// Timestamp of capability snapshot
    public let snapshotAt: Date

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

        // Estimate TFLOPS based on available processors and Apple Silicon generation
        let cores = processInfo.processorCount
        let modelID = Self.modelIdentifier()
        let estimatedTFLOPS = Self.estimateTFLOPS(cores: cores, model: modelID)

        // Battery detection
        let (battery, charging) = Self.detectBattery()

        // Compute weight incorporating thermal + battery state
        let thermalMultiplier: Double = {
            switch thermalState {
            case .nominal: return 1.0
            case .fair: return 0.75
            case .serious: return 0.4
            case .critical: return 0.0
            }
        }()

        let batteryMultiplier: Double = {
            guard let level = battery else { return 1.0 }  // Desktop Mac
            if charging { return 1.0 }  // Plugged in: full weight
            if level < 0.10 { return 0.0 }  // Critical battery: exclude
            if level < 0.20 { return 0.25 }  // Low battery: minimal work
            if level < 0.50 { return 0.6 }   // Medium battery: reduced work
            return 1.0
        }()

        let weight = min(1.0, estimatedTFLOPS / 10.0) * thermalMultiplier * batteryMultiplier

        return DeviceCapability(
            deviceID: Self.hashedDeviceID(),
            model: modelID,
            estimatedTFLOPS: estimatedTFLOPS,
            hasGPU: true,  // All Apple Silicon has GPU
            availableMemoryGB: memoryGB,
            thermalState: thermalState,
            batteryLevel: battery,
            isCharging: charging,
            computeWeight: weight,
            snapshotAt: Date()
        )
    }

    /// Whether this device is safe to schedule work on.
    public var isAvailable: Bool {
        thermalState != .critical && computeWeight > 0
    }

    /// Human-readable status summary.
    public var statusSummary: String {
        var parts: [String] = ["\(model)"]
        parts.append("\(String(format: "%.1f", estimatedTFLOPS)) TFLOPS")
        parts.append("thermal: \(thermalState.rawValue)")
        if let level = batteryLevel {
            parts.append("battery: \(Int(level * 100))%\(isCharging ? " (charging)" : "")")
        }
        parts.append("weight: \(String(format: "%.0f", computeWeight * 100))%")
        return parts.joined(separator: ", ")
    }

    // MARK: - Private Helpers

    /// Estimate TFLOPS with Apple Silicon generation awareness.
    private static func estimateTFLOPS(cores: Int, model: String) -> Double {
        let lower = model.lowercased()
        // M4 series (2024+): ~4.0 TFLOPS base
        if lower.contains("m4") { return Double(cores) * 0.25 }
        // M3 series: ~3.5 TFLOPS base
        if lower.contains("m3") { return Double(cores) * 0.22 }
        // M2 series: ~3.0 TFLOPS base
        if lower.contains("m2") { return Double(cores) * 0.19 }
        // M1 series: ~2.6 TFLOPS base
        if lower.contains("m1") { return Double(cores) * 0.16 }
        // A-series (iPhone/iPad): ~1-2 TFLOPS
        if lower.contains("iphone") || lower.contains("ipad") { return Double(cores) * 0.12 }
        // Conservative fallback
        return Double(cores) * 0.1
    }

    /// Detect battery level and charging status across macOS and iOS.
    private static func detectBattery() -> (level: Double?, isCharging: Bool) {
        #if os(iOS) || os(watchOS)
        UIDevice.current.isBatteryMonitoringEnabled = true
        let level = Double(UIDevice.current.batteryLevel)
        let charging = UIDevice.current.batteryState == .charging || UIDevice.current.batteryState == .full
        return (level >= 0 ? level : nil, charging)
        #elseif os(macOS)
        return detectMacOSBattery()
        #else
        return (nil, false)
        #endif
    }

    #if os(macOS)
    private static func detectMacOSBattery() -> (level: Double?, isCharging: Bool) {
        guard let snapshot = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(snapshot)?.takeRetainedValue() as? [Any],
              let firstSource = sources.first,
              let desc = IOPSGetPowerSourceDescription(snapshot, firstSource as CFTypeRef)?.takeUnretainedValue() as? [String: Any] else {
            return (nil, false)  // Desktop Mac without battery
        }

        let currentCap = desc[kIOPSCurrentCapacityKey] as? Int ?? 0
        let maxCap = desc[kIOPSMaxCapacityKey] as? Int ?? 100
        let isCharging = (desc[kIOPSIsChargingKey] as? Bool) ?? false
        let level = maxCap > 0 ? Double(currentCap) / Double(maxCap) : nil

        return (level, isCharging)
    }
    #endif

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
