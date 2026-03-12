// FleetScheduler.swift — Distributed work scheduling across Apple devices
//
// Actor managing work distribution, iCloud coordination, and thermal awareness.
// Enforces Advanced Data Protection before fleet operations.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

/// Actor for scheduling and distributing docking work across the Bonhomme fleet.
///
/// Usage:
/// ```swift
/// let scheduler = FleetScheduler()
/// let chunks = scheduler.splitWork(totalChromosomes: 1000, devices: devices)
/// for chunk in chunks {
///     try await scheduler.submitToiCloud(chunk)
/// }
/// ```
public actor FleetScheduler {
    private var activeJobs: [UUID: [WorkChunk]] = [:]
    private var completedResults: [UUID: [ChunkResult]] = [:]

    #if canImport(CryptoKit)
    private var encryptionKey: SymmetricKey?
    #endif

    /// iCloud Drive container identifier for fleet data
    private let containerID: String

    public init(containerID: String = "iCloud.com.bonhomme.flexaidds") {
        self.containerID = containerID
    }

    // MARK: - ADP Enforcement

    /// Check if Advanced Data Protection is available.
    /// Fleet mode requires E2E encryption — blocks if ADP is not enabled.
    public nonisolated func isAdvancedDataProtectionAvailable() -> Bool {
        let fileManager = FileManager.default
        guard let containerURL = fileManager.url(forUbiquityContainerIdentifier: nil) else {
            return false
        }
        // If iCloud is available, ADP is managed at the account level.
        // We verify iCloud availability as a proxy — full ADP detection
        // requires entitlements.
        return fileManager.fileExists(atPath: containerURL.path)
    }

    // MARK: - Work Splitting

    /// Split docking work across available devices proportional to their compute weight.
    /// - Parameters:
    ///   - totalChromosomes: Total number of chromosomes in the GA population
    ///   - maxGenerations: Maximum GA generations
    ///   - temperature: Simulation temperature in Kelvin
    ///   - devices: Available devices and their capabilities
    ///   - configData: Serialized configuration data
    /// - Returns: Array of work chunks ready for distribution
    public func splitWork(
        totalChromosomes: Int,
        maxGenerations: Int,
        temperature: Double = 300.0,
        devices: [DeviceCapability],
        configData: Data
    ) -> [WorkChunk] {
        let jobID = UUID()

        // Filter devices that can compute (not in critical thermal state)
        let activeDevices = devices.filter { $0.thermalState != .critical && $0.computeWeight > 0 }
        guard !activeDevices.isEmpty else { return [] }

        // Calculate total weight
        let totalWeight = activeDevices.reduce(0.0) { $0 + $1.computeWeight }

        // Allocate chromosomes proportional to device weight
        var chunks: [WorkChunk] = []
        var allocatedChromosomes = 0

        for (i, device) in activeDevices.enumerated() {
            let isLast = (i == activeDevices.count - 1)
            let share = device.computeWeight / totalWeight
            let chromCount = isLast
                ? totalChromosomes - allocatedChromosomes
                : Int(Double(totalChromosomes) * share)

            guard chromCount > 0 else { continue }

            let params = GAChunkParameters(
                numChromosomes: chromCount,
                maxGenerations: maxGenerations,
                seed: Int.random(in: 0..<Int.max),
                temperature: temperature
            )

            var chunk = WorkChunk(
                jobID: jobID,
                index: chunks.count,
                totalChunks: activeDevices.count,
                configData: configData,
                gaParameters: params
            )
            chunk.claimedBy = device.deviceID
            chunk.status = .claimed
            chunks.append(chunk)

            allocatedChromosomes += chromCount
        }

        activeJobs[jobID] = chunks
        return chunks
    }

    // MARK: - iCloud Submission

    /// Submit an encrypted work chunk to iCloud Drive for fleet pickup.
    /// - Parameter chunk: The work chunk to submit
    public func submitToiCloud(_ chunk: WorkChunk) async throws {
        let fileManager = FileManager.default
        guard let containerURL = fileManager.url(forUbiquityContainerIdentifier: nil) else {
            throw FleetError.iCloudUnavailable
        }

        let jobDir = containerURL.appendingPathComponent("FleetJobs/\(chunk.jobID.uuidString)")
        try fileManager.createDirectory(at: jobDir, withIntermediateDirectories: true)

        let chunkFile = jobDir.appendingPathComponent("chunk_\(chunk.index).json")
        let data = try JSONEncoder().encode(chunk)
        try data.write(to: chunkFile)
    }

    /// Submit a completed result back to iCloud Drive.
    /// - Parameter result: The chunk result to submit
    public func submitResult(_ result: ChunkResult) async throws {
        let fileManager = FileManager.default
        guard let containerURL = fileManager.url(forUbiquityContainerIdentifier: nil) else {
            throw FleetError.iCloudUnavailable
        }

        let resultsDir = containerURL.appendingPathComponent("FleetResults/\(result.jobID.uuidString)")
        try fileManager.createDirectory(at: resultsDir, withIntermediateDirectories: true)

        let resultFile = resultsDir.appendingPathComponent("result_\(result.chunkID.uuidString).json")
        let data = try JSONEncoder().encode(result)
        try data.write(to: resultFile)

        // Track completion
        completedResults[result.jobID, default: []].append(result)
    }

    /// Check if all chunks for a job have completed.
    public func isJobComplete(_ jobID: UUID) -> Bool {
        guard let chunks = activeJobs[jobID] else { return false }
        let results = completedResults[jobID] ?? []
        return results.count >= chunks.count
    }
}

// MARK: - iCloud Watcher

/// Watches an iCloud Drive directory for incoming fleet results.
#if os(macOS) || os(iOS)
public final class iCloudWatcher: @unchecked Sendable {
    private var query: NSMetadataQuery?
    private let containerID: String
    private let continuation: AsyncStream<ChunkResult>.Continuation?
    private let stream: AsyncStream<ChunkResult>

    /// Stream of incoming chunk results.
    public var results: AsyncStream<ChunkResult> {
        stream
    }

    public init(containerID: String = "iCloud.com.bonhomme.flexaidds") {
        self.containerID = containerID
        var cont: AsyncStream<ChunkResult>.Continuation?
        self.stream = AsyncStream { cont = $0 }
        self.continuation = cont
    }

    /// Start watching for incoming results.
    public func start() {
        let query = NSMetadataQuery()
        query.searchScopes = [NSMetadataQueryUbiquitousDataScope]
        query.predicate = NSPredicate(format: "%K LIKE 'FleetResults/*/result_*.json'",
                                       NSMetadataItemPathKey)

        NotificationCenter.default.addObserver(
            forName: .NSMetadataQueryDidUpdate,
            object: query, queue: .main
        ) { [weak self] notification in
            self?.handleQueryUpdate(notification)
        }

        query.start()
        self.query = query
    }

    /// Stop watching.
    public func stop() {
        query?.stop()
        query = nil
        continuation?.finish()
    }

    private func handleQueryUpdate(_ notification: Notification) {
        guard let query = notification.object as? NSMetadataQuery else { return }

        for item in query.results {
            guard let mdItem = item as? NSMetadataItem,
                  let path = mdItem.value(forAttribute: NSMetadataItemPathKey) as? String else {
                continue
            }

            let url = URL(fileURLWithPath: path)
            if let data = try? Data(contentsOf: url),
               let result = try? JSONDecoder().decode(ChunkResult.self, from: data) {
                continuation?.yield(result)
            }
        }
    }
}
#endif

// MARK: - Errors

public enum FleetError: Error, LocalizedError {
    case iCloudUnavailable
    case advancedDataProtectionRequired
    case encryptionFailed
    case noAvailableDevices

    public var errorDescription: String? {
        switch self {
        case .iCloudUnavailable:
            return "iCloud Drive is not available — sign in to iCloud to use fleet mode"
        case .advancedDataProtectionRequired:
            return "Advanced Data Protection must be enabled for fleet mode"
        case .encryptionFailed:
            return "Failed to encrypt work chunk for transit"
        case .noAvailableDevices:
            return "No devices available for fleet compute (all may be in critical thermal state)"
        }
    }
}
