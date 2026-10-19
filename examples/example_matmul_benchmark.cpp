#include "tinyinfer/ops.hpp"
#include "tinyinfer/tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tinyinfer;

namespace {

using MatmulFn = std::function<Tensor(const Tensor&, const Tensor&)>;

struct Variant {
    std::string name;
    MatmulFn run;
};

std::vector<Variant> make_variants() {
    std::vector<Variant> variants;
    variants.push_back({"ijk", matmul_ijk});
    variants.push_back({"ikj", matmul_ikj});

    for (std::size_t tile : {8, 16, 32, 64, 128, 256}) {
        variants.push_back({
            "blocked_" + std::to_string(tile),
            [tile](const Tensor& a, const Tensor& b) {
                return matmul_blocked_ikj(a, b, tile);
            }
        });
    }

    return variants;
}

std::vector<float> random_values(std::size_t count, std::mt19937& generator) {
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<float> values(count);
    for (float& value : values) {
        value = distribution(generator);
    }
    return values;
}

std::size_t iterations_for(std::size_t size) {
    if (size <= 32) {
        return 128;
    }
    if (size <= 64) {
        return 64;
    }
    if (size <= 128) {
        return 32;
    }
    if (size <= 256) {
        return 16;
    }
    return 4;
}

void verify(const Tensor& actual, const Tensor& expected) {
    if (actual.shape() != expected.shape()) {
        throw std::runtime_error("matmul result shape mismatch");
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float tolerance = 1e-4f * (1.0f + std::abs(expected[i]));
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("matmul result mismatch");
        }
    }
}

std::size_t parse_size(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid matrix size: " + text);
    }

    std::size_t consumed = 0;
    unsigned long long value = 0;

    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid matrix size: " + text);
    }

    if (consumed != text.size() || value == 0) {
        throw std::runtime_error("invalid matrix size: " + text);
    }

    return static_cast<std::size_t>(value);
}

double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::size_t> sizes;
        if (argc > 1) {
            for (int i = 1; i < argc; ++i) {
                sizes.push_back(parse_size(argv[i]));
            }
        } else {
            sizes = {32, 64, 128, 256, 512, 1024};
        }

        const auto variants = make_variants();
        std::mt19937 generator(42);
        volatile float sink = 0.0f;

#if !defined(NDEBUG)
        std::cout << "note: this is an unoptimized build; use a Release build for meaningful timings\n";
#endif
        std::cout << "matmul benchmark: 2 operations per multiply-add, times include allocation\n";

        for (std::size_t size : sizes) {
            const auto count = Tensor::element_count({size, size});
            Tensor a({size, size}, random_values(count, generator));
            Tensor b({size, size}, random_values(count, generator));

            const auto expected = matmul_ijk(a, b);
            const auto iterations = iterations_for(size);

            std::cout << "\nM = N = K = " << size << ", iterations = " << iterations << '\n';
            std::cout << std::left << std::setw(16) << "implementation"
                      << std::right << std::setw(12) << "time (ms)"
                      << std::setw(12) << "GFLOP/s" << '\n';

            for (const auto& variant : variants) {
                verify(variant.run(a, b), expected);

                const auto start = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < iterations; ++i) {
                    const Tensor result = variant.run(a, b);
                    sink = sink + result[0];
                }
                const auto elapsed = std::chrono::steady_clock::now() - start;

                const double total_seconds = std::chrono::duration<double>(elapsed).count();
                const double average_ms = milliseconds(elapsed) / static_cast<double>(iterations);
                const double gflops =
                    2 * static_cast<double>(size) * size * size * iterations / total_seconds / 1e9;

                std::cout << std::left << std::setw(16) << variant.name
                          << std::right << std::fixed << std::setprecision(3)
                          << std::setw(12) << average_ms
                          << std::setw(12) << gflops << '\n';
            }
        }

        std::cout << "\nall results verified (checksum: " << sink << ")\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
