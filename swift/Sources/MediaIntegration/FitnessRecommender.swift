// FitnessRecommender.swift — Fitness+ recommendations based on entropy state
//
// Correlates binding population entropy collapse with workout recommendations.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS

/// Recommends fitness activities based on binding population entropy state.
///
/// Maps entropy collapse / expansion to appropriate workout intensity:
/// - Collapsed entropy → gentle recovery (yoga, stretching, walking)
/// - Moderate entropy → balanced activity (cycling, swimming)
/// - High entropy → high-intensity (HIIT, running)
public struct FitnessRecommender: Sendable {

    public init() {}

    /// Generate a fitness recommendation based on entropy and health state.
    /// - Parameters:
    ///   - shannonS: Shannon configurational entropy
    ///   - hrv: HRV SDNN in ms (optional)
    ///   - sleepHours: Recent sleep duration (optional)
    /// - Returns: Fitness recommendation
    public func recommend(
        shannonS: Double,
        hrv: Double? = nil,
        sleepHours: Double? = nil
    ) -> FitnessRecommendation {
        let intensity = computeIntensity(shannonS: shannonS, hrv: hrv, sleepHours: sleepHours)
        let activities = activitiesForIntensity(intensity)

        return FitnessRecommendation(
            intensity: intensity,
            activities: activities,
            shannonS: shannonS,
            reasoning: reasoningText(intensity: intensity, shannonS: shannonS, hrv: hrv)
        )
    }

    private func computeIntensity(shannonS: Double, hrv: Double?, sleepHours: Double?) -> WorkoutIntensity {
        var score = 0.0

        // Entropy contribution (0-1)
        score += min(1.0, shannonS / 0.5) * 0.5

        // HRV contribution (higher HRV = more capacity for intensity)
        if let hrv = hrv {
            score += min(1.0, hrv / 80.0) * 0.3
        } else {
            score += 0.15  // neutral if unknown
        }

        // Sleep contribution
        if let sleep = sleepHours {
            score += min(1.0, sleep / 8.0) * 0.2
        } else {
            score += 0.1
        }

        switch score {
        case ..<0.3:  return .gentle
        case ..<0.6:  return .moderate
        default:      return .vigorous
        }
    }

    private func activitiesForIntensity(_ intensity: WorkoutIntensity) -> [String] {
        switch intensity {
        case .gentle:   return ["Yoga", "Gentle Stretching", "Walking", "Meditation"]
        case .moderate: return ["Cycling", "Swimming", "Pilates", "Light Jogging"]
        case .vigorous: return ["HIIT", "Running", "Strength Training", "Dance"]
        }
    }

    private func reasoningText(intensity: WorkoutIntensity, shannonS: Double, hrv: Double?) -> String {
        var text = "Shannon entropy S = \(String(format: "%.4f", shannonS)) kcal/mol/K"
        if shannonS < 0.1 {
            text += " (collapsed — single dominant binding mode)."
        } else if shannonS > 0.5 {
            text += " (high diversity — broad conformational search)."
        }

        if let hrv = hrv {
            text += " HRV SDNN = \(String(format: "%.0f", hrv)) ms."
            if hrv > 60 { text += " Good autonomic tone." }
            else { text += " Consider recovery." }
        }

        text += " Recommendation: \(intensity.rawValue) activity."
        return text
    }
}

// MARK: - Models

public enum WorkoutIntensity: String, Sendable, Codable {
    case gentle = "Gentle"
    case moderate = "Moderate"
    case vigorous = "Vigorous"
}

public struct FitnessRecommendation: Sendable, Codable {
    /// Recommended workout intensity
    public let intensity: WorkoutIntensity

    /// Suggested activities
    public let activities: [String]

    /// Shannon entropy value used
    public let shannonS: Double

    /// Human-readable reasoning
    public let reasoning: String
}
