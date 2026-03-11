// WorkChunk.swift — Encrypted work unit for fleet distribution
//
// Represents a chunk of docking work to be distributed across the fleet.
// Encrypted with CryptoKit before iCloud transit.
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

    /// Timestamp when the chunk was completed (nil if pending)
    public var completedAt: Date?

    /// Status of this chunk
    public var status: ChunkStatus

    public enum ChunkStatus: String, Sendable, Codable {
        case pending
        case claimed
        case running
        case completed
        case failed
    }

    public init(
        jobID: UUID, index: Int, totalChunks: Int,
        configData: Data, gaParameters: GAChunkParameters
    ) {
        self.id = UUID()
        self.jobID = jobID
        self.index = index
        self.totalChunks = totalChunks
        self.configData = configData
        self.gaParameters = gaParameters
        self.createdAt = Date()
        self.status = .pending
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

    public init(chunkID: UUID, jobID: UUID, resultData: Data, computedBy: String, computeTimeSeconds: Double) {
        self.id = UUID()
        self.chunkID = chunkID
        self.jobID = jobID
        self.resultData = resultData
        self.computedBy = computedBy
        self.computeTimeSeconds = computeTimeSeconds
        self.completedAt = Date()
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
