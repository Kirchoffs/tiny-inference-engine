#include "tinyinfer/backend.hpp"
#include "tinyinfer/tensor.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace tinyinfer;

namespace {

Tensor make_matrix(std::size_t rows, std::size_t cols, float offset) {
    std::vector<float> values(rows * cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.3679f + offset);
    }

    return Tensor({rows, cols}, std::move(values));
}

bool same(const Tensor& actual, const Tensor& expected, float tolerance = 1e-4f) {
    if (actual.shape() != expected.shape()) {
        return false;
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
            std::abs(actual[i] - expected[i]) > tolerance * (1.0f + std::abs(expected[i]))) {
            return false;
        }
    }

    return true;
}

std::size_t iterations_for(std::size_t size) {
    if (size <= 128) {
        return 8;
    }
    if (size <= 256) {
        return 4;
    }
    return 2;
}

struct Measurement {
    double milliseconds;
    double gflops;
};

Measurement measure(
    const Backend& backend,
    const Tensor& a,
    const Tensor& b,
    std::size_t iterations,
    volatile float& sink
) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        const Tensor result = backend.multiply(a, b);
        sink = sink + result[0];
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    const double seconds = std::chrono::duration<double>(elapsed).count();
    const double size = static_cast<double>(a.shape()[0]);
    return {
        std::chrono::duration<double, std::milli>(elapsed).count() / static_cast<double>(iterations),
        2 * size * size * size * static_cast<double>(iterations) / seconds / 1e9
    };
}

std::size_t parse_size(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid matrix size: " + text);
    }

    const auto value = std::stoull(text);
    if (value == 0) {
        throw std::runtime_error("matrix size must be positive");
    }

    return static_cast<std::size_t>(value);
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        const std::size_t size = argc > 1 ? parse_size(argv[1]) : 128;
        const Tensor a = make_matrix(size, size, 0.42f);
        const Tensor b = make_matrix(size, size, -1.618f);

        const CpuBackend cpu;
        const Tensor cpu_result = cpu.multiply(a, b);

        if (!metal_available()) {
            std::cout << "Metal device is not available; CPU result checksum: " << cpu_result[0] << '\n';
            return 0;
        }

        const auto metal = make_metal_backend();
        const Tensor metal_result = metal->multiply(a, b);
        if (!same(metal_result, cpu_result)) {
            throw std::runtime_error("Metal result does not match the CPU result");
        }

        const auto iterations = iterations_for(size);
        volatile float sink = 0.0f;

#if !defined(NDEBUG)
        std::cout << "note: this is an unoptimized build; use a Release build for meaningful timings\n";
#endif

        const auto cpu_measurement = measure(cpu, a, b, iterations, sink);
        const auto metal_measurement = measure(*metal, a, b, iterations, sink);

        std::cout << "matmul " << size << 'x' << size << " * " << size << 'x' << size
                  << ", iterations=" << iterations << '\n';
        std::cout << "CPU result matches Metal result\n\n";

        std::cout.setf(std::ios::fixed);
        std::cout.precision(3);
        std::cout << cpu.name() << ": " << cpu_measurement.milliseconds << " ms, "
                  << cpu_measurement.gflops << " GFLOP/s\n";
        std::cout << metal->name() << ": " << metal_measurement.milliseconds << " ms, "
                  << metal_measurement.gflops << " GFLOP/s\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
