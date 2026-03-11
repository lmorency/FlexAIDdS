// ResearchKitConsent.swift — ResearchKit consent flow for citizen science
//
// One-tap consent for contributing anonymized docking + health data.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

import Foundation

/// Consent status for citizen science data contribution.
public enum ConsentStatus: String, Sendable, Codable {
    case notAsked
    case consented
    case declined
    case withdrawn
}

/// Manages ResearchKit consent for citizen science data sharing.
///
/// When the user consents, anonymized BindingEntropyScore + health correlation
/// data can be contributed to the FlexAIDdS research database.
public struct ResearchConsent: Sendable, Codable {
    /// Current consent status
    public var status: ConsentStatus

    /// Date consent was given/declined
    public var consentDate: Date?

    /// Participant identifier (anonymized, not linked to Apple ID)
    public var participantID: String?

    /// Data categories the user consented to share
    public var consentedCategories: Set<DataCategory>

    public enum DataCategory: String, Sendable, Codable {
        case dockingResults     // Anonymized binding modes + thermodynamics
        case entropyScores      // BindingEntropyScore time series
        case healthCorrelation  // HRV/sleep correlation (fully anonymized)
    }

    public init() {
        self.status = .notAsked
        self.consentedCategories = []
    }

    /// Record consent with specified data categories.
    public mutating func giveConsent(categories: Set<DataCategory>) {
        self.status = .consented
        self.consentDate = Date()
        self.participantID = UUID().uuidString
        self.consentedCategories = categories
    }

    /// Withdraw consent.
    public mutating func withdraw() {
        self.status = .withdrawn
        self.consentedCategories = []
    }
}
