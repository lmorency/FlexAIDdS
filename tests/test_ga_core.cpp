// tests/test_ga_core.cpp — GA primitive unit tests
// Tests: QuickSort, mutate, crossover, swap_chrom, copy_chrom, genetoic/ictogene
// Apache-2.0 © 2026 Le Bonhomme Pharma

#include <gtest/gtest.h>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <vector>

#include "gaboom.h"

// ═══════════════════════════════════════════════════════════════════════
// Helper: allocate a chromosome array with genes
// ═══════════════════════════════════════════════════════════════════════

static chromosome* alloc_chroms(int n_chrom, int n_genes) {
    auto* chroms = new chromosome[n_chrom];
    for (int i = 0; i < n_chrom; ++i) {
        chroms[i].genes = new gene[n_genes];
        std::memset(chroms[i].genes, 0, sizeof(gene) * n_genes);
        chroms[i].cf = {};
        chroms[i].evalue = 0.0;
        chroms[i].app_evalue = 0.0;
        chroms[i].fitnes = 0.0;
        chroms[i].status = 'n';
    }
    return chroms;
}

static void free_chroms(chromosome* chroms, int n_chrom) {
    for (int i = 0; i < n_chrom; ++i) delete[] chroms[i].genes;
    delete[] chroms;
}

// ═══════════════════════════════════════════════════════════════════════
// QuickSort Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(QuickSortGA, SortsByEvalueAscending) {
    const int N = 10;
    const int NG = 3;
    auto* chroms = alloc_chroms(N, NG);

    // Assign descending evalues
    for (int i = 0; i < N; ++i)
        chroms[i].evalue = static_cast<double>(N - i);

    QuickSort(chroms, 0, N - 1, true);  // energy sort

    for (int i = 1; i < N; ++i)
        EXPECT_LE(chroms[i - 1].evalue, chroms[i].evalue)
            << "Not sorted ascending at index " << i;

    free_chroms(chroms, N);
}

TEST(QuickSortGA, FitnessSortPreservesData) {
    const int N = 8;
    const int NG = 2;
    auto* chroms = alloc_chroms(N, NG);

    // Assign fitness values and tag each chromosome
    double vals[] = {5.0, 3.0, 7.0, 1.0, 4.0, 6.0, 2.0, 8.0};
    for (int i = 0; i < N; ++i) {
        chroms[i].fitnes = vals[i];
        chroms[i].evalue = static_cast<double>(i);  // use evalue as ID tag
    }

    QuickSort(chroms, 0, N - 1, false);  // fitness sort

    // Verify all original values are present (sort is a permutation)
    std::vector<double> sorted_fitness;
    for (int i = 0; i < N; ++i)
        sorted_fitness.push_back(chroms[i].fitnes);
    std::sort(sorted_fitness.begin(), sorted_fitness.end());

    std::vector<double> expected(vals, vals + N);
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(sorted_fitness, expected) << "Sort must preserve all fitness values";

    free_chroms(chroms, N);
}

TEST(QuickSortGA, AlreadySorted) {
    const int N = 5;
    const int NG = 1;
    auto* chroms = alloc_chroms(N, NG);

    for (int i = 0; i < N; ++i)
        chroms[i].evalue = static_cast<double>(i);

    QuickSort(chroms, 0, N - 1, true);

    for (int i = 1; i < N; ++i)
        EXPECT_LE(chroms[i - 1].evalue, chroms[i].evalue);

    free_chroms(chroms, N);
}

TEST(QuickSortGA, SingleElement) {
    const int NG = 2;
    auto* chroms = alloc_chroms(1, NG);
    chroms[0].evalue = 42.0;

    QuickSort(chroms, 0, 0, true);
    EXPECT_EQ(chroms[0].evalue, 42.0);

    free_chroms(chroms, 1);
}

TEST(QuickSortGA, DuplicateValues) {
    const int N = 6;
    const int NG = 1;
    auto* chroms = alloc_chroms(N, NG);

    // All same evalue
    for (int i = 0; i < N; ++i)
        chroms[i].evalue = 5.0;

    QuickSort(chroms, 0, N - 1, true);

    for (int i = 0; i < N; ++i)
        EXPECT_EQ(chroms[i].evalue, 5.0);

    free_chroms(chroms, N);
}

TEST(QuickSortGA, PreservesGeneData) {
    const int N = 4;
    const int NG = 3;
    auto* chroms = alloc_chroms(N, NG);

    // Tag each chromosome with unique gene data
    for (int i = 0; i < N; ++i) {
        chroms[i].evalue = static_cast<double>(N - i);  // reverse order
        for (int g = 0; g < NG; ++g)
            chroms[i].genes[g].to_int32 = (i + 1) * 100 + g;
    }

    QuickSort(chroms, 0, N - 1, true);

    // After sorting, evalue=1.0 should have genes starting with 400
    EXPECT_EQ(chroms[0].evalue, 1.0);
    EXPECT_EQ(chroms[0].genes[0].to_int32, 400);

    free_chroms(chroms, N);
}

// ═══════════════════════════════════════════════════════════════════════
// swap_chrom Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(SwapChrom, SwapsValues) {
    const int NG = 2;
    auto* chroms = alloc_chroms(2, NG);

    chroms[0].evalue = 1.0;
    chroms[0].fitnes = 10.0;
    chroms[1].evalue = 2.0;
    chroms[1].fitnes = 20.0;

    swap_chrom(&chroms[0], &chroms[1]);

    EXPECT_EQ(chroms[0].evalue, 2.0);
    EXPECT_EQ(chroms[0].fitnes, 20.0);
    EXPECT_EQ(chroms[1].evalue, 1.0);
    EXPECT_EQ(chroms[1].fitnes, 10.0);

    free_chroms(chroms, 2);
}

// ═══════════════════════════════════════════════════════════════════════
// copy_chrom Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(CopyChrom, CopiesAllFields) {
    const int NG = 4;
    auto* src = alloc_chroms(1, NG);
    auto* dst = alloc_chroms(1, NG);

    src[0].evalue = -15.3;
    src[0].app_evalue = -12.1;
    src[0].fitnes = 42.0;
    src[0].status = 'o';
    for (int g = 0; g < NG; ++g) {
        src[0].genes[g].to_int32 = 1000 + g;
        src[0].genes[g].to_ic = 3.14 * g;
    }

    copy_chrom(dst, src, NG);

    EXPECT_EQ(dst[0].evalue, -15.3);
    EXPECT_EQ(dst[0].app_evalue, -12.1);
    EXPECT_EQ(dst[0].fitnes, 42.0);
    EXPECT_EQ(dst[0].status, 'o');
    for (int g = 0; g < NG; ++g) {
        EXPECT_EQ(dst[0].genes[g].to_int32, 1000 + g);
        EXPECT_DOUBLE_EQ(dst[0].genes[g].to_ic, 3.14 * g);
    }

    free_chroms(src, 1);
    free_chroms(dst, 1);
}

TEST(CopyChrom, DestIndependent) {
    const int NG = 2;
    auto* src = alloc_chroms(1, NG);
    auto* dst = alloc_chroms(1, NG);

    src[0].evalue = 5.0;
    src[0].genes[0].to_int32 = 999;
    copy_chrom(dst, src, NG);

    // Modify source — dest should be unchanged
    src[0].evalue = 99.0;
    src[0].genes[0].to_int32 = 0;

    EXPECT_EQ(dst[0].evalue, 5.0);
    EXPECT_EQ(dst[0].genes[0].to_int32, 999);

    free_chroms(src, 1);
    free_chroms(dst, 1);
}

// ═══════════════════════════════════════════════════════════════════════
// mutate Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(MutateGA, ZeroRateNoChange) {
    const int NG = 5;
    gene genes[NG];
    for (int g = 0; g < NG; ++g)
        genes[g].to_int32 = 12345 + g;

    int32_t originals[NG];
    for (int g = 0; g < NG; ++g)
        originals[g] = genes[g].to_int32;

    srand(42);
    mutate(genes, NG, 0.0);

    for (int g = 0; g < NG; ++g)
        EXPECT_EQ(genes[g].to_int32, originals[g])
            << "Zero mutation rate should not modify gene " << g;
}

TEST(MutateGA, FullRateChangesAll) {
    const int NG = 4;
    gene genes[NG];
    for (int g = 0; g < NG; ++g)
        genes[g].to_int32 = 0;

    srand(42);
    mutate(genes, NG, 1.0);

    // With rate=1.0, every bit is XORed with 1 => all bits set to 1
    for (int g = 0; g < NG; ++g)
        EXPECT_EQ(genes[g].to_int32, -1)  // all bits = 1 = -1 in two's complement
            << "Full mutation rate should flip all bits of gene " << g;
}

TEST(MutateGA, PartialRateModifiesSome) {
    const int NG = 10;
    gene genes[NG];
    for (int g = 0; g < NG; ++g)
        genes[g].to_int32 = 0;

    srand(42);
    mutate(genes, NG, 0.5);

    // With 50% rate across 10 genes × 32 bits = 320 bits, roughly half should flip
    int changed = 0;
    for (int g = 0; g < NG; ++g)
        if (genes[g].to_int32 != 0) changed++;

    EXPECT_GT(changed, 0) << "50% mutation rate should change at least one gene";
}

// ═══════════════════════════════════════════════════════════════════════
// crossover Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(CrossoverGA, ExchangesGenes) {
    const int NG = 6;
    gene john[NG], mary[NG];

    // Initialize with distinguishable patterns
    for (int g = 0; g < NG; ++g) {
        john[g].to_int32 = 0x11111111;  // john: all 0x1111...
        mary[g].to_int32 = 0x22222222;  // mary: all 0x2222...
        john[g].to_ic = 1.0;
        mary[g].to_ic = 2.0;
    }

    srand(42);
    crossover(john, mary, NG, 0);  // intragenes=0 (whole-gene swap)

    // After crossover, at least some gene values should have been exchanged
    bool john_has_mary_data = false;
    bool mary_has_john_data = false;
    for (int g = 0; g < NG; ++g) {
        if (john[g].to_int32 == 0x22222222) john_has_mary_data = true;
        if (mary[g].to_int32 == 0x11111111) mary_has_john_data = true;
    }

    // With 6 genes and random crossover points, exchange should happen
    // (unless both points happen to exclude all genes, which is rare)
    EXPECT_TRUE(john_has_mary_data || mary_has_john_data)
        << "Crossover should exchange genetic material between parents";
}

TEST(CrossoverGA, PreservesGeneticMaterial) {
    const int NG = 4;
    gene john[NG], mary[NG];

    // Use unique values per gene
    for (int g = 0; g < NG; ++g) {
        john[g].to_int32 = 100 + g;
        mary[g].to_int32 = 200 + g;
    }

    // Save originals
    int32_t all_values[2 * NG];
    for (int g = 0; g < NG; ++g) {
        all_values[g] = john[g].to_int32;
        all_values[NG + g] = mary[g].to_int32;
    }

    srand(42);
    crossover(john, mary, NG, 0);

    // All original values should still exist across both parents
    for (int g = 0; g < NG; ++g) {
        int32_t jv = john[g].to_int32;
        int32_t mv = mary[g].to_int32;
        bool j_found = false, m_found = false;
        for (int k = 0; k < 2 * NG; ++k) {
            if (jv == all_values[k]) j_found = true;
            if (mv == all_values[k]) m_found = true;
        }
        EXPECT_TRUE(j_found) << "john gene " << g << " has unknown value " << jv;
        EXPECT_TRUE(m_found) << "mary gene " << g << " has unknown value " << mv;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// genetoic / ictogene Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(GeneEncoding, GenetoicWithinBounds) {
    genlim gl;
    gl.min = -180.0;
    gl.max = 180.0;
    gl.del = 1.0;
    gl.nbin = 360.0;
    gl.bin = 1.0 / 360.0;

    srand(42);
    for (int i = 0; i < 100; ++i) {
        int32_t gene_val = rand();
        double ic = genetoic(&gl, gene_val);
        EXPECT_GE(ic, gl.min) << "IC below minimum";
        EXPECT_LE(ic, gl.max) << "IC above maximum";
    }
}

// ═══════════════════════════════════════════════════════════════════════
// save_snapshot Tests
// ═══════════════════════════════════════════════════════════════════════

TEST(SaveSnapshot, CopiesPopulation) {
    const int N = 3;
    const int NG = 2;
    auto* chroms = alloc_chroms(N, NG);
    auto* snapshot = alloc_chroms(N, NG);

    for (int i = 0; i < N; ++i) {
        chroms[i].evalue = -(i + 1.0);
        chroms[i].genes[0].to_int32 = i * 100;
    }

    save_snapshot(snapshot, chroms, N, NG);

    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(snapshot[i].evalue, chroms[i].evalue);
        EXPECT_EQ(snapshot[i].genes[0].to_int32, chroms[i].genes[0].to_int32);
    }

    // Modify original — snapshot should be unchanged
    chroms[0].evalue = 999.0;
    EXPECT_NE(snapshot[0].evalue, 999.0);

    free_chroms(chroms, N);
    free_chroms(snapshot, N);
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
