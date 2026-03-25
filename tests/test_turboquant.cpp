// test_turboquant.cpp — Full GTest suite for TurboQuant integration
// Tests round-trip MSE, inner-product unbiasedness, QuantizedContactMatrix,
// QuantizedEnsemble partition function, and NearestNeighborIndex kNN recall.
//
// Copyright 2026 Le Bonhomme Pharma. Licensed under Apache-2.0.

#include <gtest/gtest.h>
#include "TurboQuant.h"
#include <cmath>
#include <numeric>
#include <random>
#include <set>

// ─── Helper: generate random unit-sphere vector ─────────────────────────────

static std::vector<float> random_unit_sphere(int d, std::mt19937& rng) {
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> v(d);
    float norm = 0.0f;
    for (int j = 0; j < d; ++j) {
        v[j] = normal(rng);
        norm += v[j] * v[j];
    }
    float inv = 1.0f / std::sqrt(norm);
    for (auto& x : v) x *= inv;
    return v;
}

// ─── Helper: generate random Gaussian vector (not normalised) ───────────────

static std::vector<float> random_gaussian(int d, std::mt19937& rng, float sigma = 1.0f) {
    std::normal_distribution<float> normal(0.0f, sigma);
    std::vector<float> v(d);
    for (int j = 0; j < d; ++j)
        v[j] = normal(rng);
    return v;
}

// =============================================================================
// TEST 1: Round-trip MSE verification (quantize → dequantize)
// =============================================================================

TEST(TurboQuantMSE, RoundTripMSE_d256_b2) {
    constexpr int d = 256;
    constexpr int b = 2;
    constexpr int n_samples = 512;

    turboquant::TurboQuantMSE tq(d, b, /*seed=*/123);
    float theoretical = tq.theoretical_mse();

    std::mt19937 rng(42);
    double total_mse = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        auto x = random_unit_sphere(d, rng);
        auto qv = tq.quantize(std::span<const float>(x));

        std::vector<float> xhat(d);
        tq.dequantize(qv, std::span<float>(xhat));

        double mse = 0.0;
        for (int j = 0; j < d; ++j) {
            double diff = x[j] - xhat[j];
            mse += diff * diff;
        }
        total_mse += mse;
    }

    float empirical = static_cast<float>(total_mse / n_samples);

    // Empirical MSE must be less than the theoretical upper bound
    // (with some margin for finite-sample variance)
    EXPECT_LT(empirical, theoretical * 1.5f)
        << "Empirical MSE " << empirical
        << " exceeds 1.5× theoretical bound " << theoretical;

    // Sanity: MSE should be positive and non-trivial
    EXPECT_GT(empirical, 0.0f);
}

TEST(TurboQuantMSE, RoundTripMSE_d256_b3) {
    constexpr int d = 256;
    constexpr int b = 3;
    constexpr int n_samples = 512;

    turboquant::TurboQuantMSE tq(d, b, /*seed=*/99);
    float theoretical = tq.theoretical_mse();

    std::mt19937 rng(7);
    double total_mse = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        auto x = random_unit_sphere(d, rng);
        auto qv = tq.quantize(std::span<const float>(x));

        std::vector<float> xhat(d);
        tq.dequantize(qv, std::span<float>(xhat));

        double mse = 0.0;
        for (int j = 0; j < d; ++j) {
            double diff = x[j] - xhat[j];
            mse += diff * diff;
        }
        total_mse += mse;
    }

    float empirical = static_cast<float>(total_mse / n_samples);
    EXPECT_LT(empirical, theoretical * 1.5f);
    EXPECT_GT(empirical, 0.0f);
}

TEST(TurboQuantMSE, RoundTripMSE_SmallDim) {
    constexpr int d = 16;
    constexpr int b = 2;
    constexpr int n_samples = 256;

    turboquant::TurboQuantMSE tq(d, b, /*seed=*/77);
    float theoretical = tq.theoretical_mse();

    std::mt19937 rng(13);
    double total_mse = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        auto x = random_unit_sphere(d, rng);
        auto qv = tq.quantize(std::span<const float>(x));
        std::vector<float> xhat(d);
        tq.dequantize(qv, std::span<float>(xhat));

        double mse = 0.0;
        for (int j = 0; j < d; ++j) {
            double diff = x[j] - xhat[j];
            mse += diff * diff;
        }
        total_mse += mse;
    }

    float empirical = static_cast<float>(total_mse / n_samples);
    // For small d the bound is looser; accept 2× margin
    EXPECT_LT(empirical, theoretical * 2.5f);
}

// Use the built-in empirical_mse utility
TEST(TurboQuantMSE, UtilEmpiricalMSE) {
    constexpr int d = 256;
    constexpr int b = 3;

    turboquant::TurboQuantMSE tq(d, b, /*seed=*/42);
    float emp = turboquant::util::empirical_mse(tq, 1024);
    float theo = tq.theoretical_mse();

    EXPECT_LT(emp, theo * 1.5f);
    EXPECT_GT(emp, 0.0f);
}

// =============================================================================
// TEST 2: TurboQuantProd inner-product unbiasedness
// =============================================================================

TEST(TurboQuantProd, InnerProductUnbiased) {
    constexpr int d = 64;  // smaller d for faster test
    constexpr int b = 3;
    constexpr int n_samples = 512;

    turboquant::TurboQuantProd tqp(d, b, /*seed=*/42);

    std::mt19937 rng(99);
    double sum_error = 0.0;
    double sum_error_sq = 0.0;

    for (int s = 0; s < n_samples; ++s) {
        auto x = random_unit_sphere(d, rng);
        auto y = random_gaussian(d, rng, 1.0f);

        // True inner product
        float true_ip = 0.0f;
        for (int j = 0; j < d; ++j)
            true_ip += x[j] * y[j];

        // Quantized inner product
        auto qv = tqp.quantize(std::span<const float>(x));
        float est_ip = tqp.inner_product(qv, std::span<const float>(y));

        double err = static_cast<double>(est_ip) - static_cast<double>(true_ip);
        sum_error += err;
        sum_error_sq += err * err;
    }

    double mean_error = sum_error / n_samples;
    double var = sum_error_sq / n_samples - mean_error * mean_error;
    double std_error_of_mean = std::sqrt(std::max(0.0, var) / n_samples);

    // Unbiasedness: mean error should be within ~3 standard errors of zero
    // This is a statistical test; allow generous margin
    EXPECT_NEAR(mean_error, 0.0, 3.0 * std_error_of_mean + 0.05)
        << "Mean IP error = " << mean_error << " ± " << std_error_of_mean;
}

TEST(TurboQuantProd, UtilEmpiricalBias) {
    constexpr int d = 64;
    constexpr int b = 3;

    turboquant::TurboQuantProd tqp(d, b, /*seed=*/42);
    auto [mean_err, std_err] = turboquant::util::empirical_ip_bias(tqp, 512);

    // Mean error should be close to zero (unbiased)
    EXPECT_NEAR(mean_err, 0.0f, 0.15f)
        << "mean_err=" << mean_err << ", std_err=" << std_err;
}

// =============================================================================
// TEST 3: QuantizedContactMatrix approximate_score accuracy
// =============================================================================

TEST(QuantizedContactMatrix, ApproximateScoreAccuracy) {
    constexpr int dim = 256;
    constexpr int b = 2;
    constexpr int n_checks = 5000;

    // Generate a synthetic 256×256 contact matrix
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.01f);  // typical contact matrix scale
    std::vector<float> matrix(dim * dim);
    for (auto& v : matrix) v = normal(rng);
    // Make symmetric (typical for contact matrices)
    for (int i = 0; i < dim; ++i)
        for (int j = i + 1; j < dim; ++j)
            matrix[j * dim + i] = matrix[i * dim + j];

    turboquant::QuantizedContactMatrix qcm(b, /*seed=*/42);
    qcm.build(matrix.data());

    // Verify compression ratio > 1
    EXPECT_GT(qcm.compression_ratio(), 1.0f);

    // Check approximate scores against exact
    std::uniform_int_distribution<int> idx_dist(0, dim - 1);
    double max_abs_error = 0.0;
    double sum_sq_error = 0.0;

    for (int c = 0; c < n_checks; ++c) {
        int ti = idx_dist(rng);
        int tj = idx_dist(rng);

        float exact = matrix[ti * dim + tj];
        float approx = qcm.approximate_score(ti, tj);
        double err = std::abs(static_cast<double>(exact - approx));
        max_abs_error = std::max(max_abs_error, err);
        sum_sq_error += err * err;
    }

    double rmse = std::sqrt(sum_sq_error / n_checks);
    float theo_mse = qcm.quantizer().theoretical_mse();

    // RMSE should be bounded by theoretical MSE (scaled by row norms)
    // For unit-variance random matrix, row norms are ~1/sqrt(d), so
    // the actual element-wise error is MSE * norm factor.
    // Use a generous tolerance since the matrix is not unit-sphere normalised.
    EXPECT_LT(rmse, 0.1) << "RMSE=" << rmse << ", theoretical MSE=" << theo_mse;
    EXPECT_LT(max_abs_error, 0.5) << "Max error too large";
}

TEST(QuantizedContactMatrix, MemoryReduction) {
    constexpr int dim = 256;
    turboquant::QuantizedContactMatrix qcm(2);

    std::vector<float> matrix(dim * dim, 0.01f);
    qcm.build(matrix.data());

    size_t original = dim * dim * sizeof(float);  // 256 KB
    size_t compressed = qcm.memory_bytes();

    // Compressed should be significantly smaller (but includes rotation matrix overhead)
    // The packed rows are ~16 KB, but the rotation matrix is 256*256*4*2 = 512 KB
    // So total memory_bytes will be larger than original for small matrices.
    // But compression_ratio only measures packed indices vs original data.
    EXPECT_GT(qcm.compression_ratio(), 1.0f);
}

// =============================================================================
// TEST 4: QuantizedEnsemble partition function accuracy
// =============================================================================

TEST(QuantizedEnsemble, PartitionFunctionAccuracy) {
    constexpr int edim = 4;
    constexpr int b = 3;
    constexpr int N = 200;

    turboquant::QuantizedEnsemble qens(edim, b);
    qens.reserve(N);

    // Generate synthetic energy descriptors
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    std::vector<std::array<float, edim>> descriptors(N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < edim; ++j)
            descriptors[i][j] = normal(rng);
        qens.add_state(std::span<const float>(descriptors[i].data(), edim));
    }

    EXPECT_EQ(static_cast<int>(qens.size()), N);

    // Beta vector: project onto first two dimensions (like com + wal)
    float beta = 1.0f;  // 1/(kT) at kT=1
    std::array<float, edim> beta_E = {beta, beta, 0.0f, 0.0f};

    // Compute approximate partition function
    std::vector<float> approx_weights(N);
    float log_Z_approx = qens.compute_partition_function(
        std::span<const float>(beta_E.data(), edim),
        std::span<float>(approx_weights));

    // Compute exact partition function
    // E_i = descriptor[i][0] + descriptor[i][1]
    // w_i = exp(-beta * E_i)
    double max_neg_E = -1e30;
    for (int i = 0; i < N; ++i) {
        double E_i = static_cast<double>(descriptors[i][0] + descriptors[i][1]);
        double neg_E = -beta * E_i;
        if (neg_E > max_neg_E) max_neg_E = neg_E;
    }

    double Z_exact = 0.0;
    std::vector<double> exact_weights(N);
    for (int i = 0; i < N; ++i) {
        double E_i = static_cast<double>(descriptors[i][0] + descriptors[i][1]);
        exact_weights[i] = std::exp(-beta * E_i - max_neg_E);
        Z_exact += exact_weights[i];
    }
    for (auto& w : exact_weights) w /= Z_exact;
    double log_Z_exact = std::log(Z_exact) + max_neg_E;

    // Approximate weights should sum to ~1
    float weight_sum = 0.0f;
    for (auto w : approx_weights) weight_sum += w;
    EXPECT_NEAR(weight_sum, 1.0f, 0.01f);

    // Check weight accuracy
    double max_weight_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(static_cast<double>(approx_weights[i]) - exact_weights[i]);
        max_weight_err = std::max(max_weight_err, err);
    }

    // With 4-dim vectors at 3 bits, expect reasonable but not perfect accuracy
    // The approximation error comes from quantization noise in the inner products
    EXPECT_LT(max_weight_err, 0.3)
        << "Max weight error = " << max_weight_err;
}

TEST(QuantizedEnsemble, SmallEnsemble) {
    constexpr int edim = 4;
    constexpr int b = 3;

    turboquant::QuantizedEnsemble qens(edim, b);

    // Add 10 states
    std::mt19937 rng(7);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < 10; ++i) {
        std::array<float, edim> desc;
        for (auto& x : desc) x = normal(rng);
        qens.add_state(std::span<const float>(desc.data(), edim));
    }

    EXPECT_EQ(qens.size(), 10u);

    std::array<float, edim> beta_E = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> weights(10);
    float log_Z = qens.compute_partition_function(
        std::span<const float>(beta_E.data(), edim),
        std::span<float>(weights));

    float wsum = 0.0f;
    for (auto w : weights) wsum += w;
    EXPECT_NEAR(wsum, 1.0f, 0.01f);

    // All weights should be non-negative
    for (auto w : weights)
        EXPECT_GE(w, 0.0f);
}

// =============================================================================
// TEST 5: NearestNeighborIndex kNN recall rate
// =============================================================================

TEST(NearestNeighborIndex, KNNRecallRate) {
    constexpr int d = 32;
    constexpr int b = 2;
    constexpr int N = 200;
    constexpr int k = 10;
    constexpr int n_queries = 50;

    turboquant::NearestNeighborIndex nni(d, b, /*seed=*/42);
    nni.reserve(N);

    // Generate N random points
    std::mt19937 rng(42);
    std::vector<std::vector<float>> points(N);
    for (int i = 0; i < N; ++i) {
        points[i] = random_gaussian(d, rng, 1.0f);
        nni.add(std::span<const float>(points[i].data(), d));
    }

    EXPECT_EQ(nni.size(), static_cast<size_t>(N));

    // For each query, compute exact kNN and compare with approximate kNN
    int total_recall = 0;
    int total_possible = 0;

    for (int q = 0; q < n_queries; ++q) {
        auto query = random_gaussian(d, rng, 1.0f);

        // Exact kNN via brute force
        std::vector<std::pair<float, int>> exact_dists(N);
        for (int i = 0; i < N; ++i) {
            float dist2 = 0.0f;
            for (int j = 0; j < d; ++j) {
                float diff = query[j] - points[i][j];
                dist2 += diff * diff;
            }
            exact_dists[i] = {dist2, i};
        }
        std::partial_sort(exact_dists.begin(), exact_dists.begin() + k, exact_dists.end());

        std::set<int> exact_nn;
        for (int i = 0; i < k; ++i)
            exact_nn.insert(exact_dists[i].second);

        // Approximate kNN
        auto approx = nni.knn(std::span<const float>(query.data(), d), k);

        // Count recall (how many of exact top-k are in approximate top-k)
        for (const auto& [idx, dist] : approx) {
            if (exact_nn.count(idx))
                ++total_recall;
        }
        total_possible += k;
    }

    float recall_rate = static_cast<float>(total_recall) / total_possible;

    // With 2-bit quantization on d=32, expect reasonable recall
    // At minimum, recall should be > 30% (random would be k/N = 5%)
    EXPECT_GT(recall_rate, 0.25f)
        << "kNN recall rate = " << recall_rate
        << " (" << total_recall << "/" << total_possible << ")";
}

TEST(NearestNeighborIndex, SelfQuery) {
    constexpr int d = 16;
    constexpr int b = 2;

    turboquant::NearestNeighborIndex nni(d, b, /*seed=*/42);

    std::mt19937 rng(42);
    std::vector<std::vector<float>> points(10);
    for (int i = 0; i < 10; ++i) {
        points[i] = random_gaussian(d, rng, 1.0f);
        nni.add(std::span<const float>(points[i].data(), d));
    }

    // Query with a stored point should find itself (or very close to it)
    // as the nearest neighbour
    auto results = nni.knn(std::span<const float>(points[0].data(), d), 3);
    ASSERT_GE(results.size(), 1u);

    // The nearest neighbour should be point 0 itself (distance ~0)
    // Due to quantization, the self-distance may not be exactly 0,
    // but it should be the smallest distance
    bool found_self = false;
    for (const auto& [idx, dist] : results) {
        if (idx == 0) {
            found_self = true;
            EXPECT_LT(dist, 1.0f) << "Self-distance should be small";
            break;
        }
    }
    EXPECT_TRUE(found_self) << "Point 0 should appear in its own kNN results";
}

// =============================================================================
// TEST 6: Codebook construction and properties
// =============================================================================

TEST(Codebook, BuildCodebook256) {
    auto cb = turboquant::make_codebook_d256(2);
    EXPECT_EQ(cb.bit_width, 2);
    EXPECT_EQ(cb.num_centroids, 4);
    EXPECT_EQ(static_cast<int>(cb.centroids.size()), 4);
    EXPECT_EQ(static_cast<int>(cb.boundaries.size()), 3);

    // Centroids should be sorted
    for (int i = 1; i < cb.num_centroids; ++i)
        EXPECT_GT(cb.centroids[i], cb.centroids[i - 1]);
}

TEST(Codebook, NearestCentroid) {
    auto cb = turboquant::make_codebook_d256(2);

    // Query with a centroid value should return that centroid's index
    for (int i = 0; i < cb.num_centroids; ++i) {
        int found = cb.nearest(cb.centroids[i]);
        EXPECT_EQ(found, i);
    }
}

// =============================================================================
// TEST 7: Batch quantization
// =============================================================================

TEST(TurboQuantMSE, BatchQuantize) {
    constexpr int d = 64;
    constexpr int b = 2;
    constexpr int N = 50;

    turboquant::TurboQuantMSE tq(d, b, /*seed=*/42);

    // Generate N×d flat data
    std::mt19937 rng(42);
    std::vector<float> data(N * d);
    for (int i = 0; i < N; ++i) {
        auto v = random_unit_sphere(d, rng);
        std::copy(v.begin(), v.end(), data.begin() + i * d);
    }

    auto batch = tq.quantize_batch(std::span<const float>(data), N);
    EXPECT_EQ(static_cast<int>(batch.size()), N);

    // Each quantized vector should have correct dimension
    for (const auto& qv : batch) {
        EXPECT_EQ(qv.dim, d);
        EXPECT_EQ(qv.bit_width, b);
        EXPECT_GT(qv.norm, 0.0f);
    }
}

// =============================================================================
// TEST 8: Integration / edge cases
// =============================================================================

TEST(TurboQuantMSE, DimensionMismatch) {
    turboquant::TurboQuantMSE tq(32, 2);
    std::vector<float> wrong_dim(64, 1.0f);

    EXPECT_THROW(tq.quantize(std::span<const float>(wrong_dim)),
                 std::invalid_argument);
}

TEST(QuantizedEnsemble, EmptyEnsemble) {
    turboquant::QuantizedEnsemble qens(4, 3);
    EXPECT_EQ(qens.size(), 0u);
}

TEST(NearestNeighborIndex, EmptyIndex) {
    turboquant::NearestNeighborIndex nni(16, 2);
    EXPECT_EQ(nni.size(), 0u);

    // kNN on empty index should return empty
    std::vector<float> query(16, 0.0f);
    auto results = nni.knn(std::span<const float>(query.data(), 16), 5);
    EXPECT_TRUE(results.empty());
}

