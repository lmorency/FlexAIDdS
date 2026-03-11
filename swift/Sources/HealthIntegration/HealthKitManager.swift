// HealthKitManager.swift — HealthKit integration for FlexAIDdS
//
// Reads HRV/SDNN, sleep analysis, resting heart rate.
// Writes BindingEntropyScore as custom metadata on HK samples.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

#if canImport(HealthKit)
import HealthKit

/// Actor managing HealthKit read/write for entropy-health correlation.
@available(iOS 17.0, macOS 14.0, watchOS 10.0, *)
public actor HealthKitManager {
    private let store = HKHealthStore()

    public init() {}

    // MARK: - Authorization

    /// Request HealthKit authorization for required data types.
    public func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthError.healthDataUnavailable
        }

        let readTypes: Set<HKObjectType> = [
            HKQuantityType(.heartRateVariabilitySDNN),
            HKQuantityType(.restingHeartRate),
            HKCategoryType(.sleepAnalysis),
        ]

        // Write: we store entropy scores as workout metadata
        let writeTypes: Set<HKSampleType> = [
            HKQuantityType(.heartRateVariabilitySDNN),
        ]

        try await store.requestAuthorization(toShare: writeTypes, read: readTypes)
    }

    // MARK: - Read Health Data

    /// Read HRV (SDNN) samples from the last N hours.
    /// - Parameter hours: Number of hours to look back (default: 24)
    /// - Returns: Array of (timestamp, SDNN in ms) tuples
    public func readHRV(lastHours hours: Int = 24) async throws -> [(Date, Double)] {
        let type = HKQuantityType(.heartRateVariabilitySDNN)
        let startDate = Calendar.current.date(byAdding: .hour, value: -hours, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date())

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKQuantitySample], Error>) in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: (samples as? [HKQuantitySample]) ?? [])
                }
            }
            store.execute(query)
        }

        return samples.map { sample in
            let sdnn = sample.quantity.doubleValue(for: HKUnit.secondUnit(with: .milli))
            return (sample.startDate, sdnn)
        }
    }

    /// Read resting heart rate from the last N hours.
    public func readRestingHeartRate(lastHours hours: Int = 24) async throws -> [(Date, Double)] {
        let type = HKQuantityType(.restingHeartRate)
        let startDate = Calendar.current.date(byAdding: .hour, value: -hours, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date())

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKQuantitySample], Error>) in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: false)]
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: (samples as? [HKQuantitySample]) ?? [])
                }
            }
            store.execute(query)
        }

        return samples.map { sample in
            let bpm = sample.quantity.doubleValue(for: .count().unitDivided(by: .minute()))
            return (sample.startDate, bpm)
        }
    }

    /// Read sleep analysis from the last N hours.
    /// - Returns: Total sleep duration in hours
    public func readSleepHours(lastHours hours: Int = 24) async throws -> Double {
        let type = HKCategoryType(.sleepAnalysis)
        let startDate = Calendar.current.date(byAdding: .hour, value: -hours, to: Date())!
        let predicate = HKQuery.predicateForSamples(withStart: startDate, end: Date())

        let samples = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<[HKCategorySample], Error>) in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: nil
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: (samples as? [HKCategorySample]) ?? [])
                }
            }
            store.execute(query)
        }

        // Sum asleep durations (exclude inBed)
        let asleepSamples = samples.filter { sample in
            sample.value == HKCategoryValueSleepAnalysis.asleepCore.rawValue ||
            sample.value == HKCategoryValueSleepAnalysis.asleepDeep.rawValue ||
            sample.value == HKCategoryValueSleepAnalysis.asleepREM.rawValue
        }

        let totalSeconds = asleepSamples.reduce(0.0) { sum, sample in
            sum + sample.endDate.timeIntervalSince(sample.startDate)
        }

        return totalSeconds / 3600.0
    }

    // MARK: - Correlate

    /// Enrich a BindingEntropyScore with the latest HealthKit data.
    public func enrich(_ score: BindingEntropyScore) async throws -> BindingEntropyScore {
        var enriched = score

        if let latestHRV = try await readHRV(lastHours: 4).first {
            enriched.hrvSDNN = latestHRV.1
        }

        if let latestHR = try await readRestingHeartRate(lastHours: 4).first {
            enriched.restingHeartRate = latestHR.1
        }

        enriched.sleepHours = try await readSleepHours(lastHours: 24)

        return enriched
    }
}

// MARK: - Errors

public enum HealthError: Error, LocalizedError {
    case healthDataUnavailable
    case authorizationDenied

    public var errorDescription: String? {
        switch self {
        case .healthDataUnavailable:
            return "HealthKit is not available on this device"
        case .authorizationDenied:
            return "HealthKit authorization was denied — enable in Settings"
        }
    }
}
#endif
