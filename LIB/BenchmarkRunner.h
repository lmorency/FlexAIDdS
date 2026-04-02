// =============================================================================
// BenchmarkRunner.h — Benchmark timing utilities for FlexAIDdS
//
// Copyright 2026 Le Bonhomme Pharma. Licensed under Apache-2.0.
// =============================================================================

#pragma once

#include <chrono>

namespace bench {

class Timer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }
    void stop()  { stop_  = std::chrono::steady_clock::now(); }

    double elapsed_s() const {
        return std::chrono::duration<double>(stop_ - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point stop_{};
};

} // namespace bench
