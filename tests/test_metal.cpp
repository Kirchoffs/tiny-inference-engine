#include "tinyinfer_test_framework.hpp"

#include "tinyinfer/backend.hpp"
#include "tinyinfer/graph.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tinyinfer;
using namespace tinyinfer_test;

namespace {

std::unique_ptr<Backend> require_metal() {
    if (!metal_available()) {
        skip("Metal device is not available");
    }

    return make_metal_backend();
}

Tensor make_matrix(std::size_t rows, std::size_t cols, float offset) {
    std::vector<float> values(rows * cols);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = std::sin(static_cast<float>(i) * 0.37f + offset);
    }

    return Tensor({rows, cols}, std::move(values));
}

} // anonymous namespace

TINYINFER_TEST(metal, backend_reports_name) {
    const auto backend = require_metal();
    check(std::string(backend->name()).find("Metal") != std::string::npos, "unexpected backend name");
}

TINYINFER_TEST(metal, multiply_matches_cpu) {
    const auto backend = require_metal();
    const Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});

    equal(backend->multiply(a, b), matmul_ijk(a, b), 1e-4f);
}

TINYINFER_TEST(metal, multiply_handles_rectangular_sizes) {
    const auto backend = require_metal();
    const Tensor a = make_matrix(17, 13, 0.5f);
    const Tensor b = make_matrix(13, 19, -0.25f);

    equal(backend->multiply(a, b), matmul_ijk(a, b), 1e-4f);
}

TINYINFER_TEST(metal, graph_uses_metal_backend) {
    const auto backend = require_metal();

    Graph graph;
    auto x = graph.input("x", {2, 3});
    auto w = graph.constant(Tensor({3, 2}, {1, 2, -1, 0, 2, -2}));
    auto b = graph.constant(Tensor({2}, {0.5f, -0.5f}));
    auto hidden = graph.relu(graph.bias(graph.multiply(x, w), b));
    auto output = graph.softmax(hidden);

    const Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor weights({3, 2}, {1, 2, -1, 0, 2, -2});
    const Tensor bias({2}, {0.5f, -0.5f});
    const auto expected = softmax(relu(tinyinfer::bias(matmul_ijk(a, weights), bias)));

    equal(
        graph.run(output, std::unordered_map<std::string, Tensor>{{"x", a}}, *backend),
        expected,
        1e-4f
    );
}

TINYINFER_TEST(metal, rejects_invalid_shapes) {
    const auto backend = require_metal();

    rejects(
        [&] { backend->multiply(Tensor({2, 3}), Tensor({2, 3})); },
        "expected inner dimension rejection"
    );
    rejects(
        [&] { backend->multiply(Tensor({2}), Tensor({2, 2})); },
        "expected rank rejection"
    );
}
