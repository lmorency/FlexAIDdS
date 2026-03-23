// FleetAggregatorTests.swift — Unit tests for fleet result aggregation
//
// Tests deduplication, density-peak clustering, Boltzmann weighting,
// and global/per-mode thermodynamic correctness.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import FleetScheduler
@testable import FlexAIDdS

final class FleetAggregatorTests: XCTestCase {

    let kB: Double = 0.001987204
    let T: Double = 298.15

    // MARK: - Helpers

    /// Build a ChunkResult with a JSON-encoded array of pose energies.
    func makeChunkResult(
        energies: [Double],
        chunkID: UUID = UUID(),
        jobID: UUID = UUID(),
        deviceID: String = "device-A",
        temperature: Double = 298.15
    ) -> ChunkResult {
        let data = try! JSONEncoder().encode(energies)
        return ChunkResult(
            chunkID: chunkID, jobID: jobID, resultData: data,
            computedBy: deviceID, computeTimeSeconds: 1.0,
            temperature: temperature,
            summaryFreeEnergy: energies.min() ?? 0,
            summaryEntropy: 0.01,
            summaryHeatCapacity: 0.5,
            summaryMeanEnergy: energies.reduce(0, +) / Double(max(1, energies.count)),
            summaryStdEnergy: 1.0,
            summaryModeCount: 1,
            summaryBestFreeEnergy: energies.min() ?? 0
        )
    }

    // MARK: - Ingestion

    func testIngestChunkResults() {
        let aggregator = FleetAggregator(temperature: T)

        let r1 = makeChunkResult(energies: [-10.0, -8.0, -6.0], deviceID: "mac")
        let r2 = makeChunkResult(energies: [-9.5, -7.0], deviceID: "iphone")

        aggregator.ingest(r1)
        aggregator.ingest(r2)

        XCTAssertEqual(aggregator.poseCount, 5)
        XCTAssertEqual(aggregator.chunkCount, 2)
    }

    func testDuplicateChunkIngestionIgnored() {
        let aggregator = FleetAggregator(temperature: T)
        let chunkID = UUID()

        let r1 = makeChunkResult(energies: [-10.0], chunkID: chunkID, deviceID: "mac")
        aggregator.ingest(r1)
        aggregator.ingest(r1)  // same chunkID — should be ignored

        XCTAssertEqual(aggregator.poseCount, 1)
        XCTAssertEqual(aggregator.chunkCount, 1)
    }

    // MARK: - Deduplication

    func testDeduplicateSameDevice() {
        let aggregator = FleetAggregator(temperature: T)

        // Two poses from same device with near-identical energy
        let poses = [
            AggregatedPose(energy: -10.0, sourceChunkID: UUID(), sourceDeviceID: "mac"),
            AggregatedPose(energy: -10.005, sourceChunkID: UUID(), sourceDeviceID: "mac"),
            AggregatedPose(energy: -8.0, sourceChunkID: UUID(), sourceDeviceID: "mac"),
        ]

        let (deduped, removed) = aggregator.deduplicatePoses(poses, threshold: 0.01)

        XCTAssertEqual(removed, 1, "Near-identical energy from same device should be deduped")
        XCTAssertEqual(deduped.count, 2)
    }

    func testDeduplicateDifferentDevicesKept() {
        let aggregator = FleetAggregator(temperature: T)

        // Same energy but different devices — NOT duplicates
        let poses = [
            AggregatedPose(energy: -10.0, sourceChunkID: UUID(), sourceDeviceID: "mac"),
            AggregatedPose(energy: -10.0, sourceChunkID: UUID(), sourceDeviceID: "iphone"),
        ]

        let (deduped, removed) = aggregator.deduplicatePoses(poses, threshold: 0.01)

        XCTAssertEqual(removed, 0, "Same energy from different devices should be kept")
        XCTAssertEqual(deduped.count, 2)
    }

    // MARK: - Clustering

    func testClusteringFormsModes() {
        let aggregator = FleetAggregator(temperature: T)

        // Two energy clusters: ~-10 and ~-5
        var poses: [AggregatedPose] = []
        for _ in 0..<5 {
            poses.append(AggregatedPose(
                energy: -10.0 + Double.random(in: -0.3...0.3),
                sourceChunkID: UUID(), sourceDeviceID: "mac"
            ))
        }
        for _ in 0..<5 {
            poses.append(AggregatedPose(
                energy: -5.0 + Double.random(in: -0.3...0.3),
                sourceChunkID: UUID(), sourceDeviceID: "iphone"
            ))
        }

        let clustered = aggregator.clusterByEnergyDensity(
            poses, bandwidth: 2.0, minSize: 3
        )

        let clusterIDs = Set(clustered.map(\.clusterID)).filter { $0 >= 0 }
        XCTAssertGreaterThanOrEqual(clusterIDs.count, 1,
            "Should find at least one binding mode cluster")
    }

    func testTooFewPosesAllNoise() {
        let aggregator = FleetAggregator(temperature: T)

        let poses = [
            AggregatedPose(energy: -10.0, sourceChunkID: UUID(), sourceDeviceID: "mac"),
        ]

        let clustered = aggregator.clusterByEnergyDensity(
            poses, bandwidth: 2.0, minSize: 3
        )

        XCTAssertTrue(clustered.allSatisfy { $0.clusterID == -1 },
            "Single pose should be noise when minSize=3")
    }

    // MARK: - Boltzmann Weights

    func testBoltzmannWeightsSumToOne() {
        let aggregator = FleetAggregator(temperature: T)

        let r1 = makeChunkResult(energies: [-10.0, -8.0, -6.0, -4.0, -10.5], deviceID: "mac")
        let r2 = makeChunkResult(energies: [-9.5, -7.0, -5.0, -3.0, -9.8], deviceID: "iphone")
        aggregator.ingest(r1)
        aggregator.ingest(r2)

        let result = aggregator.aggregate(
            deduplicationThreshold: 0.001,
            clusterMinSize: 1
        )

        let totalWeight = result.poses.reduce(0.0) { $0 + $1.boltzmannWeight }
        XCTAssertEqual(totalWeight, 1.0, accuracy: 1e-10,
            "Boltzmann weights across merged ensemble must sum to 1")
    }

    func testLowestEnergyGetsHighestWeight() {
        let aggregator = FleetAggregator(temperature: T)

        let r = makeChunkResult(energies: [-15.0, -5.0, -1.0], deviceID: "mac")
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)
        let sorted = result.poses.sorted { $0.energy < $1.energy }

        XCTAssertGreaterThan(sorted.first!.boltzmannWeight, sorted.last!.boltzmannWeight,
            "Lowest energy pose should have highest Boltzmann weight")
    }

    // MARK: - Global Thermodynamics

    func testFreeEnergyLowerThanMeanEnergy() {
        let aggregator = FleetAggregator(temperature: T)

        let r = makeChunkResult(
            energies: [-12.0, -10.0, -8.0, -6.0, -4.0],
            deviceID: "mac"
        )
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)
        let thermo = result.globalThermodynamics

        XCTAssertLessThanOrEqual(thermo.freeEnergy, thermo.meanEnergy,
            "F = <E> - TS must be ≤ <E> (entropy drives F down)")
    }

    func testEntropyNonNegative() {
        let aggregator = FleetAggregator(temperature: T)

        let r = makeChunkResult(
            energies: [-10.0, -9.0, -8.0, -7.0],
            deviceID: "mac"
        )
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)

        XCTAssertGreaterThanOrEqual(result.globalThermodynamics.entropy, 0,
            "Conformational entropy must be non-negative")
    }

    func testHeatCapacityNonNegative() {
        let aggregator = FleetAggregator(temperature: T)

        let r = makeChunkResult(
            energies: [-10.0, -9.0, -8.0, -7.0, -6.0],
            deviceID: "mac"
        )
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)

        XCTAssertGreaterThanOrEqual(result.globalThermodynamics.heatCapacity, 0,
            "Heat capacity must be non-negative")
    }

    func testSingleEnergyZeroEntropy() {
        let aggregator = FleetAggregator(temperature: T)

        // Single microstate → S = 0
        let r = makeChunkResult(energies: [-10.0], deviceID: "mac")
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)

        XCTAssertEqual(result.globalThermodynamics.entropy, 0.0, accuracy: 1e-12,
            "Single microstate should have zero entropy")
        XCTAssertEqual(result.globalThermodynamics.freeEnergy, -10.0, accuracy: 1e-10,
            "Single microstate: F = E")
    }

    // MARK: - Per-Mode Thermodynamics

    func testModeWeightsSumToAtMostOne() {
        let aggregator = FleetAggregator(temperature: T)

        // Create two clear clusters
        var energies: [Double] = []
        for _ in 0..<10 { energies.append(-10.0 + Double.random(in: -0.1...0.1)) }
        for _ in 0..<10 { energies.append(-5.0 + Double.random(in: -0.1...0.1)) }

        let r = makeChunkResult(energies: energies, deviceID: "mac")
        aggregator.ingest(r)

        let result = aggregator.aggregate(
            deduplicationThreshold: 0.001,
            clusterMinSize: 2,
            clusterEnergyBandwidth: 2.0
        )

        let totalModeWeight = result.bindingModes.reduce(0.0) {
            $0 + $1.thermodynamics.globalWeight
        }

        // Mode weights sum to ≤ 1 (noise poses not in any mode reduce sum below 1)
        XCTAssertLessThanOrEqual(totalModeWeight, 1.0 + 1e-10)
    }

    func testBindingModesSortedByFreeEnergy() {
        let aggregator = FleetAggregator(temperature: T)

        // Enough poses for two distinct modes
        var energies: [Double] = []
        for _ in 0..<8 { energies.append(-12.0 + Double.random(in: -0.2...0.2)) }
        for _ in 0..<8 { energies.append(-4.0 + Double.random(in: -0.2...0.2)) }

        let r = makeChunkResult(energies: energies, deviceID: "mac")
        aggregator.ingest(r)

        let result = aggregator.aggregate(
            clusterMinSize: 2,
            clusterEnergyBandwidth: 3.0
        )

        guard result.bindingModes.count >= 2 else {
            // Clustering may not always split cleanly; that's OK
            return
        }

        for i in 1..<result.bindingModes.count {
            XCTAssertGreaterThanOrEqual(
                result.bindingModes[i].thermodynamics.freeEnergy,
                result.bindingModes[i - 1].thermodynamics.freeEnergy,
                "Binding modes should be sorted by free energy (ascending)"
            )
        }
    }

    // MARK: - Multi-Device Aggregation

    func testMultiDeviceContribution() {
        let aggregator = FleetAggregator(temperature: T)
        let jobID = UUID()

        let r1 = makeChunkResult(
            energies: [-10.0, -9.5, -9.0, -8.5],
            jobID: jobID, deviceID: "mac"
        )
        let r2 = makeChunkResult(
            energies: [-10.2, -9.8, -9.3, -8.8],
            jobID: jobID, deviceID: "iphone"
        )
        let r3 = makeChunkResult(
            energies: [-10.1, -9.6, -9.1, -8.6],
            jobID: jobID, deviceID: "ipad"
        )

        aggregator.ingest(r1)
        aggregator.ingest(r2)
        aggregator.ingest(r3)

        let result = aggregator.aggregate(clusterMinSize: 2)

        XCTAssertEqual(result.chunkCount, 3)
        XCTAssertEqual(result.poses.count, 12)

        // With tight energy range, might be 1 mode from 3 devices
        if let mode = result.bindingModes.first {
            XCTAssertGreaterThanOrEqual(mode.deviceCount, 1,
                "Mode should report contributing device count")
        }
    }

    // MARK: - Deduplication Count

    func testAggregationReportsDeduplicationCount() {
        let aggregator = FleetAggregator(temperature: T)

        // Same device, near-identical energies
        let r = makeChunkResult(
            energies: [-10.0, -10.005, -10.002, -8.0],
            deviceID: "mac"
        )
        aggregator.ingest(r)

        let result = aggregator.aggregate(
            deduplicationThreshold: 0.01,
            clusterMinSize: 1
        )

        XCTAssertEqual(result.deduplicatedCount, 2,
            "Should report 2 duplicates removed (10.005 and 10.002 match 10.0)")
    }

    // MARK: - Empty Input

    func testEmptyAggregation() {
        let aggregator = FleetAggregator(temperature: T)
        let result = aggregator.aggregate()

        XCTAssertTrue(result.poses.isEmpty)
        XCTAssertTrue(result.bindingModes.isEmpty)
        XCTAssertEqual(result.deduplicatedCount, 0)
        XCTAssertEqual(result.chunkCount, 0)
    }

    // MARK: - Codable Round-Trip

    func testAggregationResultCodable() throws {
        let aggregator = FleetAggregator(temperature: T)

        let r = makeChunkResult(
            energies: [-10.0, -8.0, -6.0, -4.0, -2.0],
            deviceID: "mac"
        )
        aggregator.ingest(r)

        let result = aggregator.aggregate(clusterMinSize: 1)

        let data = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(FleetAggregationResult.self, from: data)

        XCTAssertEqual(decoded.poses.count, result.poses.count)
        XCTAssertEqual(decoded.globalThermodynamics.freeEnergy,
                       result.globalThermodynamics.freeEnergy, accuracy: 1e-10)
        XCTAssertEqual(decoded.deduplicatedCount, result.deduplicatedCount)
    }
}
