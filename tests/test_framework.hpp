#pragma once

#include "tinyinfer/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tinyinfer_test {

struct TestCase {
    std::string suite;
    std::string name;
    std::function<void()> body;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct TestRegistrar {
    TestRegistrar(std::string suite, std::string name, std::function<void()> body) {
        registry().push_back({std::move(suite), std::move(name), std::move(body)});
    }
};

inline void check(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class F>
void rejects(F&& f, const char* message = "expected exception") {
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }

    check(caught, message);
}

inline void close(float actual, float expected, float tolerance = 1e-5f) {
    check(
        std::isfinite(actual) && 
        std::isfinite(expected) &&
        std::abs(actual - expected) <= tolerance * (1.0f + std::abs(expected)),
        "numerical mismatch"
    );
}

inline void equal(const tinyinfer::Tensor& actual, const tinyinfer::Tensor& expected, float tolerance = 1e-5f) {
    check(actual.shape() == expected.shape(), "shape mismatch");

    for (std::size_t i = 0; i < actual.size(); ++i) {
        close(actual[i], expected[i], tolerance);
    }
}

inline int run_all(const std::string& suite = {}) {
    int passed = 0;
    int failed = 0;
    int skipped = 0;

    for (const auto& test : registry()) {
        if (!suite.empty() && test.suite != suite) {
            ++skipped;
            continue;
        }

        try {
            test.body();
            ++passed;
            std::cout << "[ PASS ] " << test.suite << '.' << test.name << '\n';
        } catch (const std::exception& e) {
            ++failed;
            std::cerr << "[ FAIL ] " << test.suite << '.' << test.name << ": " << e.what() << '\n';
        } catch (...) {
            ++failed;
            std::cerr << "[ FAIL ] " << test.suite << '.' << test.name << ": unknown exception\n";
        }
    }

    if (!suite.empty() && passed + failed == 0) {
        std::cerr << "no tests found for suite '" << suite << "'\n";
        return 1;
    }

    std::cout << '\n' << passed << " passed, " << failed << " failed";
    if (skipped != 0) {
        std::cout << ", " << skipped << " skipped";
    }
    std::cout << '\n';

    return failed == 0 ? 0 : 1;
}

} // namespace tinyinfer_test

#define TINYINFER_TEST(suite, name)                                                          \
    static void tinyinfer_test_##suite##_##name();                                           \
    static const ::tinyinfer_test::TestRegistrar tinyinfer_test_registrar_##suite##_##name { \
        #suite, #name, tinyinfer_test_##suite##_##name                                       \
    };                                                                                       \
    static void tinyinfer_test_##suite##_##name()
