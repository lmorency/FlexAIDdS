// ga_diversity.h — GA population diversity monitoring and entropy collapse mitigation
//
// Tracks Shannon entropy of allele frequency distributions across all gene
// dimensions. When population diversity drops below a critical threshold
// (entropy collapse), triggers catastrophic mutations to re-diversify.
//
// Integration: called from gaboom.cpp after the entropy convergence check.
//
// Copyright 2024-2026 Louis-Philippe Morency / NRGlab, Universite de Montreal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// Requires gaboom.h to be included first (for chromosome, genlim types)
#include "gaboom.h"

namespace ga_diversity {

struct DiversityMetrics {
    double allele_entropy;     // mean Shannon entropy across all genes (nats)
    double min_gene_entropy;   // worst-case single gene entropy (nats)
    bool   collapse_detected;  // true if allele_entropy < threshold
};

// Compute the Shannon entropy of allele frequency distributions across all
// gene dimensions. For each gene, allele values are binned into n_bins
// and the Shannon entropy is computed from the resulting histogram.
//
// Returns a DiversityMetrics struct with the mean entropy, minimum per-gene
// entropy, and whether collapse was detected (entropy below threshold).
inline DiversityMetrics compute_diversity(
    const chromosome* pop,
    int num_chrom,
    int num_genes,
    const genlim* gene_lim,
    double collapse_threshold,
    int n_bins = 20)
{
    DiversityMetrics result = { 0.0, 1e30, false };
    if (num_chrom <= 0 || num_genes <= 0) {
        result.collapse_detected = true;
        return result;
    }

    double total_entropy = 0.0;
    double max_possible = std::log(static_cast<double>(n_bins)); // max entropy

    for (int g = 0; g < num_genes; ++g) {
        double gmin = gene_lim[g].min;
        double gmax = gene_lim[g].max;
        double range = gmax - gmin;
        if (range <= 0.0) range = 1.0; // degenerate gene

        // Bin the allele values
        std::vector<int> counts(n_bins, 0);
        for (int c = 0; c < num_chrom; ++c) {
            double val = pop[c].genes[g].to_ic;
            int bin = static_cast<int>((val - gmin) / range * n_bins);
            if (bin < 0) bin = 0;
            if (bin >= n_bins) bin = n_bins - 1;
            counts[bin]++;
        }

        // Compute Shannon entropy for this gene
        double H = 0.0;
        double n = static_cast<double>(num_chrom);
        for (int b = 0; b < n_bins; ++b) {
            if (counts[b] > 0) {
                double p = static_cast<double>(counts[b]) / n;
                H -= p * std::log(p);
            }
        }

        // Normalize to [0, 1]
        double H_norm = (max_possible > 0.0) ? H / max_possible : 0.0;
        total_entropy += H_norm;
        if (H_norm < result.min_gene_entropy)
            result.min_gene_entropy = H_norm;
    }

    result.allele_entropy = total_entropy / static_cast<double>(num_genes);
    result.collapse_detected = (result.allele_entropy < collapse_threshold);
    return result;
}

// Apply catastrophic mutation to the worst-fitness fraction of the population.
// Randomly re-initializes genes for the selected individuals within their
// gene limit bounds.
inline void catastrophic_mutation(
    chromosome* pop,
    int num_chrom,
    int num_genes,
    const genlim* gene_lim,
    double fraction,
    std::mt19937& rng)
{
    if (num_chrom <= 0 || fraction <= 0.0) return;

    // Population is assumed to be sorted by fitness (best first).
    // Mutate the bottom fraction (worst fitness individuals).
    int n_mutate = static_cast<int>(std::ceil(fraction * num_chrom));
    if (n_mutate > num_chrom) n_mutate = num_chrom;

    int start = num_chrom - n_mutate;

    for (int c = start; c < num_chrom; ++c) {
        for (int g = 0; g < num_genes; ++g) {
            double gmin = gene_lim[g].min;
            double gmax = gene_lim[g].max;
            std::uniform_real_distribution<double> dist(gmin, gmax);
            pop[c].genes[g].to_ic = dist(rng);
        }
        pop[c].status = 'r'; // mark as requiring re-evaluation
    }
}

} // namespace ga_diversity
