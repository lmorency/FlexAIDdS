// FitnessRecommender.swift — Fitness+ recommendations based on entropy state
//
// Correlates binding population entropy collapse with workout recommendations.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation
import FlexAIDdS
import HealthIntegration

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

    /// Generate a fitness recommendation using decomposed entropy data.
    ///
    /// When `BindingEntropyScore` is available, uses convergence status and
    /// vibrational dominance to refine the intensity recommendation:
    /// - Not converged → gentle (uncertain state, recovery preferred)
    /// - Vibrational dominance (S_vib >> S_conf) → moderate ("flow state")
    /// - Mode imbalance (kinetic trapping) → vigorous (energizing)
    public func recommend(entropyScore: BindingEntropyScore) -> FitnessRecommendation {
        let intensity = computeDecomposedIntensity(score: entropyScore)
        let activities = activitiesForIntensity(intensity)

        var reasoning = "Shannon entropy S = \(String(format: "%.4f", entropyScore.shannonS)) kcal/mol/K"
        if let decomp = entropyScore.shannonDecomposition {
            reasoning += ", S_conf = \(String(format: "%.4f", decomp.configurational)) nats"
            reasoning += ", S_vib = \(String(format: "%.6f", decomp.vibrational)) kcal/mol/K"
            if !decomp.isConverged {
                reasoning += ". Entropy not converged — calming activity recommended while sampling continues."
            }
        }
        reasoning += " Recommendation: \(intensity.rawValue) activity."

        return FitnessRecommendation(
            intensity: intensity,
            activities: activities,
            shannonS: entropyScore.shannonS,
            reasoning: reasoning
        )
    }

    private func computeDecomposedIntensity(score: BindingEntropyScore) -> WorkoutIntensity {
        guard let decomp = score.shannonDecomposition else {
            return computeIntensity(shannonS: score.shannonS, hrv: score.hrvSDNN, sleepHours: score.sleepHours)
        }

        // Not converged → gentle (uncertain state)
        if !decomp.isConverged {
            return .gentle
        }

        // Vibrational dominance → moderate "flow state"
        let kB = 0.001987206
        let sConfPhysical = decomp.configurational * kB
        if decomp.vibrational > sConfPhysical * 3.0 && decomp.vibrational > 0.001 {
            return .moderate
        }

        // Mode imbalance (kinetic trapping suspected) → vigorous
        if decomp.perModeEntropy.count >= 2 {
            if let maxS = decomp.perModeEntropy.max(),
               let minS = decomp.perModeEntropy.filter({ $0 > 0 }).min(),
               maxS > minS * 10 {
                return .vigorous
            }
        }

        // Fall back to scalar-based intensity
        return computeIntensity(shannonS: score.shannonS, hrv: score.hrvSDNN, sleepHours: score.sleepHours)
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
