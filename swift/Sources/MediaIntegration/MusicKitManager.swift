// MusicKitManager.swift — MusicKit integration for entropy-driven playlists
//
// Generates playlist recommendations based on shannonS + HRV state.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

#if canImport(MusicKit)
import MusicKit

/// Actor managing MusicKit playlist generation based on docking entropy state.
///
/// Maps Shannon entropy collapse / expansion to musical tempo and genre:
/// - High entropy (diverse population) → upbeat, energetic music
/// - Low entropy (collapsed population) → calm, focused music
/// - Entropy recovery → progressive buildups
@available(iOS 17.0, macOS 14.0, *)
public actor MusicKitManager {

    public init() {}

    // MARK: - Authorization

    /// Request MusicKit authorization.
    /// - Returns: Authorization status
    @discardableResult
    public func requestAuthorization() async -> MusicAuthorization.Status {
        await MusicAuthorization.request()
    }

    // MARK: - Playlist Generation

    /// Generate a playlist recommendation based on entropy and health state.
    /// - Parameters:
    ///   - shannonS: Shannon configurational entropy
    ///   - hrv: Heart rate variability SDNN (ms), if available
    ///   - limit: Maximum number of songs (default: 20)
    /// - Returns: Collection of recommended songs
    public func generatePlaylist(
        shannonS: Double,
        hrv: Double? = nil,
        limit: Int = 20
    ) async throws -> MusicItemCollection<Song> {
        let mood = EntropyMood.from(shannonS: shannonS, hrv: hrv)

        var request = MusicCatalogSearchRequest(term: mood.searchTerm, types: [Song.self])
        request.limit = limit

        let response = try await request.response()
        return response.songs
    }

    /// Get a curated search term based on entropy-health state.
    public nonisolated func moodDescription(shannonS: Double, hrv: Double? = nil) -> String {
        let mood = EntropyMood.from(shannonS: shannonS, hrv: hrv)
        return mood.description
    }
}

// MARK: - Entropy Mood Mapping

/// Maps entropy + HRV state to a musical mood.
enum EntropyMood {
    case collapsed       // Very low entropy — single dominant binding mode
    case converging      // Entropy decreasing — population focusing
    case balanced        // Moderate entropy — healthy exploration
    case expanding       // High entropy — broad conformational search
    case recovery        // Low entropy + improving HRV — system recovering

    var searchTerm: String {
        switch self {
        case .collapsed:  return "ambient meditation calm"
        case .converging: return "focus deep concentration"
        case .balanced:   return "chill electronic lofi"
        case .expanding:  return "upbeat energetic dance"
        case .recovery:   return "gentle progressive build"
        }
    }

    var description: String {
        switch self {
        case .collapsed:  return "Entropy collapsed — calm, meditative"
        case .converging: return "Population converging — focused, deep"
        case .balanced:   return "Balanced ensemble — chill, steady"
        case .expanding:  return "High entropy — energetic, diverse"
        case .recovery:   return "Entropy recovering — gentle progression"
        }
    }

    static func from(shannonS: Double, hrv: Double?) -> EntropyMood {
        // Entropy thresholds (kcal mol^-1 K^-1)
        let lowThreshold = 0.05
        let highThreshold = 0.5

        if shannonS < lowThreshold {
            if let hrv = hrv, hrv > 50 {
                return .recovery  // Low entropy but good HRV = recovering
            }
            return .collapsed
        } else if shannonS < 0.2 {
            return .converging
        } else if shannonS < highThreshold {
            return .balanced
        } else {
            return .expanding
        }
    }
}
#endif
