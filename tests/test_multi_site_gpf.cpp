// test_multi_site_gpf.cpp — Unit tests for MultiSiteGPF
//
// Verifies multi-site grand partition function: independent sites,
// cooperativity, cross-site selectivity, and thread safety.
//
// Copyright 2026 Le Bonhomme Pharma
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "MultiSiteGPF.h"
#include "statmech.h"

#include <cmath>
#include <thread>

using namespace target;

static constexpr double kT_300 = statmech::kB_kcal * 300.0;

// ════════════════════════════════════════════════════════════════════════
// Construction & site management
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, EmptyConstruction) {
    MultiSiteGPF msgpf(300.0);
    EXPECT_EQ(msgpf.num_sites(), 0);
    EXPECT_NEAR(msgpf.log_Xi(), 0.0, 1e-12);
    EXPECT_NEAR(msgpf.empty_probability(), 1.0, 1e-12);
}

TEST(MultiSiteGPF, AddSites) {
    MultiSiteGPF msgpf(300.0);
    int s0 = msgpf.add_site("orthosteric");
    int s1 = msgpf.add_site("allosteric");

    EXPECT_EQ(s0, 0);
    EXPECT_EQ(s1, 1);
    EXPECT_EQ(msgpf.num_sites(), 2);
    EXPECT_EQ(msgpf.site_index("orthosteric"), 0);
    EXPECT_EQ(msgpf.site_index("allosteric"), 1);
    EXPECT_EQ(msgpf.site_index("nonexistent"), -1);
}

TEST(MultiSiteGPF, DuplicateSiteThrows) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("active");
    EXPECT_THROW(msgpf.add_site("active"), std::invalid_argument);
}

TEST(MultiSiteGPF, InvalidSiteAccess) {
    MultiSiteGPF msgpf(300.0);
    EXPECT_THROW(msgpf.site(0), std::invalid_argument);
    EXPECT_THROW(msgpf.add_ligand(0, "A", 5.0), std::invalid_argument);
}

// ════════════════════════════════════════════════════════════════════════
// Independent sites (no cooperativity)
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, IndependentTwoSites) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("orthosteric");
    msgpf.add_site("allosteric");

    // Different ligands at different sites
    msgpf.add_ligand(0, "drug_A", 10.0);
    msgpf.add_ligand(1, "modulator_B", 8.0);

    // Each site's occupancy should be independent
    double occ_ortho = msgpf.site_occupancy(0);
    double occ_allo  = msgpf.site_occupancy(1);

    EXPECT_GT(occ_ortho, 0.99);   // strong binder
    EXPECT_GT(occ_allo, 0.99);    // strong binder

    // log Ξ = ln Ξ_0 + ln Ξ_1
    double log_xi_0 = msgpf.site(0).log_Xi();
    double log_xi_1 = msgpf.site(1).log_Xi();
    EXPECT_NEAR(msgpf.log_Xi(), log_xi_0 + log_xi_1, 1e-10);

    // Binding probabilities at each site
    EXPECT_GT(msgpf.binding_probability(0, "drug_A"), 0.99);
    EXPECT_GT(msgpf.binding_probability(1, "modulator_B"), 0.99);

    // Cross-site: drug_A not at allosteric site
    EXPECT_FALSE(msgpf.site(1).has_ligand("drug_A"));
}

TEST(MultiSiteGPF, SameLigandMultipleSites) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("site_A");
    msgpf.add_site("site_B");

    // Same ligand at both sites with different affinities
    msgpf.add_ligand(0, "drug", 10.0);   // strong at site A
    msgpf.add_ligand(1, "drug", 2.0);    // weak at site B

    double p_A = msgpf.binding_probability(0, "drug");
    double p_B = msgpf.binding_probability(1, "drug");

    EXPECT_GT(p_A, p_B);  // prefers site A

    // Cross-site analysis
    auto analysis = msgpf.cross_site_analysis();
    ASSERT_EQ(analysis.size(), 1u);
    EXPECT_EQ(analysis[0].ligand_name, "drug");
    EXPECT_EQ(analysis[0].best_site_idx, 0);  // site A is best
}

TEST(MultiSiteGPF, IndependentProbabilitiesMultiply) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("S1");
    msgpf.add_site("S2");

    msgpf.add_ligand(0, "A", 10.0);
    msgpf.add_ligand(1, "B", 10.0);

    // p(both empty) = p(S1 empty) * p(S2 empty)
    double p_empty_1 = msgpf.site(0).empty_probability();
    double p_empty_2 = msgpf.site(1).empty_probability();
    double p_both_empty = msgpf.empty_probability();

    EXPECT_NEAR(p_both_empty, p_empty_1 * p_empty_2, 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Cooperativity
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, PositiveCooperativity) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("orthosteric");
    msgpf.add_site("allosteric");

    msgpf.add_ligand(0, "drug", 5.0);     // moderate affinity
    msgpf.add_ligand(1, "modulator", 5.0); // moderate affinity

    // Baseline: independent
    double p_drug_indep = msgpf.binding_probability(0, "drug");

    // Set positive cooperativity
    msgpf.set_cooperativity(0, 1, 5.0);  // ω = 5

    // Binding probability should increase with positive cooperativity
    double p_drug_coop = msgpf.binding_probability(0, "drug");
    EXPECT_GT(p_drug_coop, p_drug_indep);
}

TEST(MultiSiteGPF, NegativeCooperativity) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("orthosteric");
    msgpf.add_site("allosteric");

    msgpf.add_ligand(0, "drug", 5.0);
    msgpf.add_ligand(1, "modulator", 5.0);

    double p_drug_indep = msgpf.binding_probability(0, "drug");

    // Set negative cooperativity
    msgpf.set_cooperativity(0, 1, 0.2);  // ω = 0.2

    // With negative cooperativity, the marginal probability model
    // reduces probability when other site is occupied
    // Note: this is an approximation — full model would require enumeration
    double p_drug_neg = msgpf.binding_probability(0, "drug");
    EXPECT_LT(p_drug_neg, p_drug_indep);
}

TEST(MultiSiteGPF, CooperativitySymmetric) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("A");
    msgpf.add_site("B");

    msgpf.set_cooperativity(0, 1, 3.0);
    EXPECT_NEAR(msgpf.cooperativity(0, 1), 3.0, 1e-12);
    EXPECT_NEAR(msgpf.cooperativity(1, 0), 3.0, 1e-12);  // symmetric
}

TEST(MultiSiteGPF, InvalidCooperativityThrows) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("A");
    msgpf.add_site("B");
    EXPECT_THROW(msgpf.set_cooperativity(0, 1, 0.0), std::invalid_argument);
    EXPECT_THROW(msgpf.set_cooperativity(0, 1, -1.0), std::invalid_argument);
}

TEST(MultiSiteGPF, DefaultCooperativityIsOne) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("A");
    msgpf.add_site("B");
    EXPECT_NEAR(msgpf.cooperativity(0, 1), 1.0, 1e-12);
}

// ════════════════════════════════════════════════════════════════════════
// Selectivity at specific sites
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, SiteSpecificSelectivity) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("active");

    msgpf.add_ligand(0, "strong", 20.0);
    msgpf.add_ligand(0, "weak", 2.0);

    EXPECT_NEAR(msgpf.selectivity(0, "strong", "weak"),
                std::exp(18.0), std::exp(18.0) * 1e-10);
}

// ════════════════════════════════════════════════════════════════════════
// Cross-site analysis
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, CrossSiteAnalysis) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("kinase_domain");
    msgpf.add_site("regulatory");

    // Ligand X binds both sites, prefers kinase domain
    msgpf.add_ligand(0, "X", 15.0);  // strong at kinase
    msgpf.add_ligand(1, "X", 3.0);   // weak at regulatory

    // Ligand Y only at regulatory
    msgpf.add_ligand(1, "Y", 10.0);

    auto analysis = msgpf.cross_site_analysis();
    ASSERT_EQ(analysis.size(), 2u);

    // Find X and Y in results
    const MultiSiteGPF::CrossSiteSelectivity* x_res = nullptr;
    const MultiSiteGPF::CrossSiteSelectivity* y_res = nullptr;
    for (const auto& r : analysis) {
        if (r.ligand_name == "X") x_res = &r;
        if (r.ligand_name == "Y") y_res = &r;
    }
    ASSERT_NE(x_res, nullptr);
    ASSERT_NE(y_res, nullptr);

    // X prefers kinase domain (site 0)
    EXPECT_EQ(x_res->best_site_idx, 0);
    EXPECT_GT(x_res->site_probabilities[0], x_res->site_probabilities[1]);

    // Y only at regulatory (site 1)
    EXPECT_EQ(y_res->best_site_idx, 1);
    EXPECT_NEAR(y_res->site_probabilities[0], 0.0, 1e-12);  // not at kinase
}

// ════════════════════════════════════════════════════════════════════════
// Concurrent multi-site registration
// ════════════════════════════════════════════════════════════════════════

TEST(MultiSiteGPF, ConcurrentSiteRegistration) {
    MultiSiteGPF msgpf(300.0);
    msgpf.add_site("S0");
    msgpf.add_site("S1");

    const int N = 50;
    auto worker = [&](int id) {
        int site = id % 2;  // alternate between sites
        msgpf.add_ligand(site, "lig_" + std::to_string(id),
                         static_cast<double>(id));
    };

    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i)
        threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();

    EXPECT_EQ(msgpf.site(0).num_ligands(), 25);  // even IDs → site 0
    EXPECT_EQ(msgpf.site(1).num_ligands(), 25);  // odd IDs → site 1

    // Total log Ξ should be finite
    EXPECT_TRUE(std::isfinite(msgpf.log_Xi()));
}
