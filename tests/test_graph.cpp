#include "test_framework.hpp"

#include "tinyinfer/graph.hpp"
#include "tinyinfer/tensor.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tinyinfer;
using namespace tinyinfer_test;

namespace {

using Feeds = std::unordered_map<std::string, Tensor>;

} // anonymous namespace

TINYINFER_TEST(graph, constant_output) {
    Graph graph;
    auto vector = graph.constant(Tensor({2}, {1, 2}));
    auto scalar = graph.constant(Tensor({}, 3.14f));

    const Feeds no_feeds;
    equal(graph.run(vector, no_feeds), Tensor({2}, {1, 2}));
    equal(graph.run(scalar, no_feeds), Tensor({}, 3.14f));
}

TINYINFER_TEST(graph, input_passes_through) {
    Graph graph;
    auto x = graph.input("x", {2, 2});

    equal(
        graph.run(x, Feeds{{"x", Tensor({2, 2}, {1, 2, 3, 4})}}),
        Tensor({2, 2}, {1, 2, 3, 4})
    );
}

TINYINFER_TEST(graph, mlp_matches_expected) {
    Graph graph;
    auto x = graph.input("x", {2, 2});
    auto w1 = graph.constant(Tensor({2, 3}, {-1, -1, -1, 1, 1, 1}));
    auto b1 = graph.constant(Tensor({3}, {1, 1, 1}));
    auto hidden = graph.activation(graph.bias(graph.multiply(x, w1), b1));
    auto w2 = graph.constant(Tensor({3, 2}, {2, 1, -1, -1, 2, 1}));
    auto b2 = graph.constant(Tensor({2}, {-3, -1}));
    auto logits = graph.bias(graph.multiply(hidden, w2), b2);
    auto probabilities = graph.probabilities(logits);

    const Feeds feeds{{"x", Tensor({2, 2}, {1, 2, 2, 4})}};
    equal(graph.run(hidden, feeds), Tensor({2, 3}, {2, 2, 2, 3, 3, 3}));
    equal(graph.run(logits, feeds), Tensor({2, 2}, {3, 1, 6, 2}));
    equal(
        graph.run(probabilities, feeds),
        Tensor({2, 2}, {0.8807971f, 0.1192029f, 0.9820138f, 0.0179862f})
    );
}

TINYINFER_TEST(graph, uses_injected_matmul_kernel) {
    Graph graph;
    auto x = graph.input("x", {1, 2});
    auto weights = graph.constant(Tensor({2, 2}, {2, 1, -1, -2}));
    auto output = graph.multiply(x, weights);

    int calls = 0;
    MatMulKernel kernel = [&](const Tensor& a, const Tensor& b) {
        ++calls;
        check(a.size() == 2, "unexpected left operand");
        check(b.size() == 4, "unexpected right operand");
        return Tensor({1, 2}, {55, -55}); // deliberately not the real result, to prove the kernel 
    };

    equal(graph.run(output, Feeds{{"x", Tensor({1, 2}, {5, 5})}}, kernel), Tensor({1, 2}, {55, -55}));
    check(calls == 1, "matmul kernel should be called once");
}

TINYINFER_TEST(graph, reuses_graph_across_runs) {
    Graph graph;
    auto x = graph.input("x", {1, 3});
    auto weights = graph.constant(Tensor({3, 1}, {1, 1, 1}));
    auto bias = graph.constant(Tensor({1}, 1.618f));
    auto output = graph.bias(graph.multiply(x, weights), bias);

    equal(graph.run(output, Feeds{{"x", Tensor({1, 3}, {1, 2, 3})}}), Tensor({1, 1}, 7.618f));
    equal(graph.run(output, Feeds{{"x", Tensor({1, 3}, {4, 5, 6})}}), Tensor({1, 1}, 16.618f));
}

TINYINFER_TEST(graph, skips_nodes_not_needed_for_output) {
    Graph graph;
    auto x_unused = graph.input("x_unused", {1, 2});
    auto weights_unused = graph.constant(Tensor({2, 2}, {2, 1, 1, 2}));
    auto output_unused = graph.multiply(x_unused, weights_unused);
    (void) output_unused;
    auto x = graph.input("x", {1, 2});
    auto weights = graph.constant(Tensor({2, 2}, {2, -1, -1, 2}));
    auto output = graph.multiply(x, weights);

    int calls = 0;
    MatMulKernel kernel = [&](const Tensor& a, const Tensor& b) {
        ++calls;
        return matmul(a, b);
    };

    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 2}, {5, 6})}}, kernel),
        Tensor({1, 2}, {4, 7})
    );
    check(calls == 1, "only the live matmul should run");
}

TINYINFER_TEST(graph, rejects_invalid_inputs) {
    rejects([] { Graph graph; graph.input("", {2}); }, "expected empty name rejection");

    rejects(
        [] {
            Graph graph;
            graph.input("x", {2});
            graph.input("x", {2});
        },
        "expected duplicate name rejection"
    );

    rejects([] { Graph graph; graph.input("x", {0, 1}); }, "expected zero dimension rejection");

    rejects(
        [] {
            Graph graph;
            graph.input("x", {std::numeric_limits<std::size_t>::max(), 2});
        },
        "expected element count overflow rejection"
    );
}

TINYINFER_TEST(graph, rejects_invalid_dependencies) {
    rejects([] { Graph graph; graph.multiply(0, 1); }, "expected invalid dependency rejection");

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.activation(x + 1);
        },
        "expected future dependency rejection"
    );
}

TINYINFER_TEST(graph, rejects_invalid_runs) {
    rejects(
        [] {
            Graph graph;
            const Feeds no_feeds;
            graph.run(0, no_feeds);
        },
        "expected invalid output rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            const Feeds no_feeds;
            graph.run(x, no_feeds);
        },
        "expected missing feed rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.run(x, Feeds{{"x", Tensor({1, 2})}});
        },
        "expected feed shape rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.run(x, Feeds{{"x", Tensor({2}, {1, 2})}}, MatMulKernel{});
        },
        "expected empty kernel rejection"
    );
}
