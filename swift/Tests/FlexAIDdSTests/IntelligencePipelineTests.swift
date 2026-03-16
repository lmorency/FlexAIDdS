// IntelligencePipelineTests.swift — Tests for unified IntelligencePipeline
//
// Tests that IntelligencePipeline correctly orchestrates all rule-based
// features, isolates failures, and produces correct aggregate results.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import XCTest
@testable import FlexAIDdS
@testable import Intelligence

final class IntelligencePipelineTests: XCTestCase {

    // MARK: - Factories

    private func makeBindingMode(
        freeEnergy: Double = -8.5,
        entropy: Double = 0.005,
        enthalpy: Double = -10.0,
        heatCapacity: Double = 0.02,
        size: Int = 15
    ) -> BindingModeResult {
        BindingModeResult(
            freeEnergy: freeEnergy,
            entropy: entropy,
            enthalpy: enthalpy,
            heatCapacity: heatCapacity,
            size: size,
            poses: (0..<size).map { i in
                PoseResult(cf: freeEnergy + Double(i) * 0.5, reachDist: Float(i) * 0.3)
            }
        )
    }

    private func makeDockingResult(modeCount: Int = 2) -> DockingResult {
        let modes = (0..<modeCount).map { i in
            makeBindingMode(freeEnergy: -8.5 + Double(i) * 2.0)
        }
        return DockingResult(
            bindingModes: modes,
            globalThermodynamics: ThermodynamicResult(
                temperature: 298.15, logZ: 10.0, freeEnergy: -8.0,
                meanEnergy: -9.5, meanEnergySq: 92.0,
                heatCapacity: 0.03, entropy: 0.006, stdEnergy: 1.2
            ),
            temperature: 298.15,
            populationSize: 500
        )
    }

    private func makeGASnapshot() -> GAProgressSnapshot {
        GAProgressSnapshot(
            currentGeneration: 150,
            maxGenerations: 200,
            bestFitness: -12.5,
            meanFitness: -8.3,
            populationDiversity: 0.65,
            generationsSinceImprovement: 60,
            fitnessTrajectory: [-10.0, -11.0, -11.5, -12.0, -12.5],
            diversityTrajectory: [0.8, 0.75, 0.7, 0.68, 0.65],
            isImproving: false,
            isDiversityCollapsed: false,
            populationSize: 300
        )
    }

    private func makeCleftFeatures() -> CleftFeatures {
        CleftFeatures(
            volume: 500, depth: 7.0, sphereCount: 25,
            maxSphereRadius: 3.5, hydrophobicFraction: 0.55,
            anchorResidueCount: 6, elongation: 0.4, solventExposure: 0.25
        )
    }

    // MARK: - Pipeline Tests

    func testPipelineGeneratesModeNarrative() {
        let pipeline = IntelligencePipeline()
        let result = pipeline.analyze(dockingResult: makeDockingResult())

        XCTAssertNotNil(result.modeNarrative)
        XCTAssertEqual(result.modeNarrative?.modeDescriptions.count, 2)
        XCTAssertTrue(result.errors.isEmpty, "No errors expected: \(result.errors)")
    }

    func testPipelineGeneratesPoseQuality() {
        let pipeline = IntelligencePipeline()
        let result = pipeline.analyze(dockingResult: makeDockingResult())

        XCTAssertFalse(result.poseQualityReports.isEmpty, "Should have pose quality reports")
    }

    func testPipelineWithConvergenceCoaching() {
        let pipeline = IntelligencePipeline()
        let result = pipeline.analyze(
            dockingResult: makeDockingResult(),
            gaSnapshot: makeGASnapshot()
        )

        XCTAssertNotNil(result.convergenceCoaching)
        // Stagnation > 25% with isImproving=false → stopEarly
        XCTAssertEqual(result.convergenceCoaching?.advice, .stopEarly)
    }

    func testPipelineWithCleftAssessment() {
        let pipeline = IntelligencePipeline()
        let result = pipeline.analyze(
            dockingResult: makeDockingResult(),
            cleftFeatures: makeCleftFeatures()
        )

        XCTAssertNotNil(result.cleftAssessment)
        XCTAssertEqual(result.cleftAssessment?.druggability, .high)
    }

    func testPipelineWithEmptyBindingModes() {
        let pipeline = IntelligencePipeline()
        let empty = DockingResult(
            bindingModes: [],
            globalThermodynamics: ThermodynamicResult(
                temperature: 298.15, logZ: 0, freeEnergy: 0,
                meanEnergy: 0, meanEnergySq: 0,
                heatCapacity: 0, entropy: 0, stdEnergy: 0
            ),
            temperature: 298.15,
            populationSize: 0
        )
        let result = pipeline.analyze(dockingResult: empty)

        XCTAssertNil(result.modeNarrative)
        XCTAssertTrue(result.errors.contains { $0.contains("BindingModeNarrator") })
    }

    func testPipelineIsolatesFailures() {
        let pipeline = IntelligencePipeline()
        let empty = DockingResult(
            bindingModes: [],
            globalThermodynamics: ThermodynamicResult(
                temperature: 298.15, logZ: 0, freeEnergy: 0,
                meanEnergy: 0, meanEnergySq: 0,
                heatCapacity: 0, entropy: 0, stdEnergy: 0
            ),
            temperature: 298.15,
            populationSize: 0
        )
        // Even with empty modes, convergence coaching should still work
        let result = pipeline.analyze(
            dockingResult: empty,
            gaSnapshot: makeGASnapshot()
        )

        XCTAssertNil(result.modeNarrative, "Mode narrative should fail for empty modes")
        XCTAssertNotNil(result.convergenceCoaching, "Convergence coaching should succeed independently")
    }

    // MARK: - Codable Round-Trip

    func testIntelligenceResultCodable() throws {
        let pipeline = IntelligencePipeline()
        let original = pipeline.analyze(
            dockingResult: makeDockingResult(),
            gaSnapshot: makeGASnapshot(),
            cleftFeatures: makeCleftFeatures()
        )

        let data = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(IntelligenceResult.self, from: data)

        XCTAssertEqual(decoded.modeNarrative?.modeDescriptions.count,
                       original.modeNarrative?.modeDescriptions.count)
        XCTAssertEqual(decoded.convergenceCoaching?.advice,
                       original.convergenceCoaching?.advice)
        XCTAssertEqual(decoded.cleftAssessment?.druggability,
                       original.cleftAssessment?.druggability)
        XCTAssertEqual(decoded.poseQualityReports.count,
                       original.poseQualityReports.count)
        XCTAssertEqual(decoded.errors.count, original.errors.count)
    }
}
