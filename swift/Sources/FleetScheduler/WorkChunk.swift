// WorkChunk.swift — Encrypted work unit for fleet distribution
//
// Represents a chunk of docking work to be distributed across the fleet.
// Encrypted with CryptoKit before iCloud transit.
// Includes timeout, retry logic, and orphan recovery support.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
#if canImport(CryptoKit)
import CryptoKit
#endif

/// A chunk of docking work to be distributed across the fleet.
public struct WorkChunk: Sendable, Codable, Identifiable {
    /// Unique identifier for this chunk
    public let id: UUID

    /// Job ID this chunk belongs to
    public let jobID: UUID

    /// Chunk index within the job
    public let index: Int

    /// Total number of chunks in the job
    public let totalChunks: Int

    /// Configuration data for this chunk (serialized)
    public let configData: Data

    /// GA parameter overrides (e.g., subset of chromosomes)
    public let gaParameters: GAChunkParameters

    /// Device ID that claimed this chunk (nil if unclaimed)
    public var claimedBy: String?

    /// Timestamp when the chunk was created
    public let createdAt: Date

    /// Timestamp when the chunk was claimed (for timeout detection)
    public var claimedAt: Date?

    /// Timestamp when the chunk was completed (nil if pending)
    public var completedAt: Date?

    /// Status of this chunk
    public var status: ChunkStatus

    /// Number of times this chunk has been retried
    public var retryCount: Int

    /// Maximum number of retries before permanent failure
    public let maxRetries: Int

    /// Timeout in seconds — chunk is orphaned if not completed within this window
    public let timeoutSeconds: TimeInterval

    /// Parent chunk ID (set when a chunk was split from a larger chunk)
    public var parentChunkID: UUID?

    /// Priority level (higher = scheduled first)
    public var priority: ChunkPriority

    public enum ChunkStatus: String, Sendable, Codable {
        case pending
        case claimed
        case running
        case completed
        case failed
        case orphaned        // Timed out, eligible for reclaim
        case permanentlyFailed  // Exceeded max retries
    }

    public enum ChunkPriority: Int, Sendable, Codable, Comparable {
        case low = 0
        case normal = 1
        case high = 2      // Retried chunks get elevated priority
        case critical = 3  // Final retry attempt

        public static func < (lhs: ChunkPriority, rhs: ChunkPriority) -> Bool {
            lhs.rawValue < rhs.rawValue
        }
    }

    public init(
        jobID: UUID, index: Int, totalChunks: Int,
        configData: Data, gaParameters: GAChunkParameters,
        maxRetries: Int = 3, timeoutSeconds: TimeInterval = 3600
    ) {
        self.id = UUID()
        self.jobID = jobID
        self.index = index
        self.totalChunks = totalChunks
        self.configData = configData
        self.gaParameters = gaParameters
        self.createdAt = Date()
        self.status = .pending
        self.retryCount = 0
        self.maxRetries = maxRetries
        self.timeoutSeconds = timeoutSeconds
        self.priority = .normal
    }

    // MARK: - Timeout & Retry

    /// Whether this chunk has exceeded its timeout and should be reclaimed.
    public var isTimedOut: Bool {
        guard let claimed = claimedAt else { return false }
        return Date().timeIntervalSince(claimed) > timeoutSeconds
    }

    /// Whether this chunk can be retried.
    public var canRetry: Bool {
        retryCount < maxRetries
    }

    /// Mark this chunk as orphaned and prepare for retry.
    /// Returns false if max retries exceeded (becomes permanently failed).
    public mutating func markOrphanedForRetry() -> Bool {
        retryCount += 1
        claimedBy = nil
        claimedAt = nil

        if retryCount >= maxRetries {
            status = .permanentlyFailed
            return false
        }

        status = .orphaned
        // Elevate priority on retry so it gets picked up faster
        priority = retryCount >= maxRetries - 1 ? .critical : .high
        return true
    }

    /// Claim this chunk for a device.
    public mutating func claim(by deviceID: String) {
        claimedBy = deviceID
        claimedAt = Date()
        status = .claimed
    }

    /// Mark this chunk as running.
    public mutating func markRunning() {
        status = .running
    }

    /// Mark this chunk as completed.
    public mutating func markCompleted() {
        status = .completed
        completedAt = Date()
    }

    /// Mark this chunk as failed (device-reported failure, not timeout).
    public mutating func markFailed() {
        if canRetry {
            _ = markOrphanedForRetry()
        } else {
            status = .permanentlyFailed
        }
    }

    /// Wall time since claim (for progress tracking).
    public var elapsedSinceClaim: TimeInterval? {
        guard let claimed = claimedAt else { return nil }
        return Date().timeIntervalSince(claimed)
    }
}

/// GA parameters for a specific chunk of work.
public struct GAChunkParameters: Sendable, Codable {
    /// Number of chromosomes for this chunk
    public let numChromosomes: Int

    /// Maximum generations for this chunk
    public let maxGenerations: Int

    /// Random seed (unique per chunk for diversity)
    public let seed: Int

    /// Temperature in Kelvin
    public let temperature: Double

    public init(numChromosomes: Int, maxGenerations: Int, seed: Int, temperature: Double = 300.0) {
        self.numChromosomes = numChromosomes
        self.maxGenerations = maxGenerations
        self.seed = seed
        self.temperature = temperature
    }
}

/// Result from a completed work chunk.
public struct ChunkResult: Sendable, Codable, Identifiable {
    public let id: UUID
    public let chunkID: UUID
    public let jobID: UUID

    /// Serialized docking result data
    public let resultData: Data

    /// Device that computed this chunk
    public let computedBy: String

    /// Computation time in seconds
    public let computeTimeSeconds: Double

    /// Timestamp of completion
    public let completedAt: Date

    /// Thermal state of device at completion (for fleet health tracking)
    public let thermalStateAtCompletion: String?

    /// Battery level at completion (for fleet health tracking)
    public let batteryLevelAtCompletion: Double?

    public init(
        chunkID: UUID, jobID: UUID, resultData: Data,
        computedBy: String, computeTimeSeconds: Double,
        thermalState: String? = nil, batteryLevel: Double? = nil
    ) {
        self.id = UUID()
        self.chunkID = chunkID
        self.jobID = jobID
        self.resultData = resultData
        self.computedBy = computedBy
        self.computeTimeSeconds = computeTimeSeconds
        self.completedAt = Date()
        self.thermalStateAtCompletion = thermalState
        self.batteryLevelAtCompletion = batteryLevel
    }
}

// MARK: - Encryption

#if canImport(CryptoKit)
extension WorkChunk {
    /// Encrypt this chunk for iCloud transit using ChaChaPoly.
    /// - Parameter key: Symmetric key for encryption
    /// - Returns: Encrypted data (nonce + ciphertext + tag)
    public func encrypt(using key: SymmetricKey) throws -> Data {
        let encoded = try JSONEncoder().encode(self)
        let sealed = try ChaChaPoly.seal(encoded, using: key)
        return sealed.combined
    }

    /// Decrypt a work chunk from encrypted data.
    /// - Parameters:
    ///   - data: Encrypted data
    ///   - key: Symmetric key for decryption
    /// - Returns: Decrypted WorkChunk
    public static func decrypt(from data: Data, using key: SymmetricKey) throws -> WorkChunk {
        let box = try ChaChaPoly.SealedBox(combined: data)
        let decrypted = try ChaChaPoly.open(box, using: key)
        return try JSONDecoder().decode(WorkChunk.self, from: decrypted)
    }
}
#endif
