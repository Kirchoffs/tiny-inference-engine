#include "tinyinfer_test_framework.hpp"

#include "tinyinfer/backend.hpp"
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

const CpuBackend& cpu_backend() {
    static const CpuBackend backend;
    return backend;
}

class CountingBackend final : public Backend {
public:
    mutable int multiply_calls = 0;

    const char* name() const override {
        return "counting";
    }

    Tensor multiply(const Tensor& a, const Tensor& b) const override {
        ++multiply_calls;
        return matmul_ijk(a, b);
    }
};

} // anonymous namespace

TINYINFER_TEST(graph, constant_output) {
    Graph graph;
    auto vector = graph.constant(Tensor({2}, {1, 2}));
    auto scalar = graph.constant(Tensor({}, 3.14f));

    const Feeds no_feeds;
    equal(graph.run(vector, no_feeds, cpu_backend()), Tensor({2}, {1, 2}));
    equal(graph.run(scalar, no_feeds, cpu_backend()), Tensor({}, 3.14f));
}

TINYINFER_TEST(graph, input_passes_through) {
    Graph graph;
    auto x = graph.input("x", {2, 2});

    equal(
        graph.run(x, Feeds{{"x", Tensor({2, 2}, {1, 2, 3, 4})}}, cpu_backend()),
        Tensor({2, 2}, {1, 2, 3, 4})
    );
}

TINYINFER_TEST(graph, mlp_matches_expected) {
    Graph graph;
    auto x = graph.input("x", {2, 2});
    auto w1 = graph.constant(Tensor({2, 3}, {-1, -1, -1, 1, 1, 1}));
    auto b1 = graph.constant(Tensor({3}, {1, 1, 1}));
    auto hidden = graph.relu(graph.bias(graph.multiply(x, w1), b1));
    auto w2 = graph.constant(Tensor({3, 2}, {2, 1, -1, -1, 2, 1}));
    auto b2 = graph.constant(Tensor({2}, {-3, -1}));
    auto logits = graph.bias(graph.multiply(hidden, w2), b2);
    auto probabilities = graph.softmax(logits);

    const Feeds feeds{{"x", Tensor({2, 2}, {1, 2, 2, 4})}};
    equal(graph.run(hidden, feeds, cpu_backend()), Tensor({2, 3}, {2, 2, 2, 3, 3, 3}));
    equal(graph.run(logits, feeds, cpu_backend()), Tensor({2, 2}, {3, 1, 6, 2}));
    equal(
        graph.run(probabilities, feeds, cpu_backend()),
        Tensor({2, 2}, {0.8807971f, 0.1192029f, 0.9820138f, 0.0179862f})
    );
}

TINYINFER_TEST(graph, uses_backend_for_matmul) {
    Graph graph;
    auto x = graph.input("x", {1, 2});
    auto weights = graph.constant(Tensor({2, 2}, {1, 0, 0, 1}));
    auto output = graph.multiply(x, weights);

    class RecordingBackend final : public Backend {
    public:
        mutable int calls = 0;

        const char* name() const override {
            return "recording";
        }

        Tensor multiply(const Tensor&, const Tensor&) const override {
            ++calls;
            return Tensor({1, 2}, {7, 8});
        }
    };

    const RecordingBackend backend;
    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 2}, {3, 4})}}, backend),
        Tensor({1, 2}, {7, 8})
    );
    check(backend.calls == 1, "the backend should handle the matmul node");
}

TINYINFER_TEST(graph, uses_backend_for_all_ops) {
    Graph graph;
    auto x = graph.input("x", {1, 1});
    auto w = graph.constant(Tensor({1, 1}, 1.0f));
    auto b = graph.constant(Tensor({1}, 1.0f));
    auto output = graph.softmax(graph.relu(graph.bias(graph.multiply(x, w), b)));

    class AllOpsBackend final : public Backend {
    public:
        mutable int multiply_calls = 0;
        mutable int bias_calls = 0;
        mutable int relu_calls = 0;
        mutable int softmax_calls = 0;

        const char* name() const override {
            return "all-ops";
        }

        Tensor multiply(const Tensor&, const Tensor&) const override {
            ++multiply_calls;
            return Tensor({1, 1}, 1.0f);
        }

        Tensor bias(const Tensor&, const Tensor&) const override {
            ++bias_calls;
            return Tensor({1, 1}, 2.0f);
        }

        Tensor relu(const Tensor&) const override {
            ++relu_calls;
            return Tensor({1, 1}, 3.0f);
        }

        Tensor softmax(const Tensor&) const override {
            ++softmax_calls;
            return Tensor({1, 1}, 4.0f);
        }
    };

    const AllOpsBackend backend;
    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 1}, 0.0f)}}, backend),
        Tensor({1, 1}, 4.0f)
    );
    check(backend.multiply_calls == 1, "multiply should use the backend");
    check(backend.bias_calls == 1, "bias should use the backend");
    check(backend.relu_calls == 1, "relu should use the backend");
    check(backend.softmax_calls == 1, "softmax should use the backend");
}

TINYINFER_TEST(graph, reuses_graph_across_runs) {
    Graph graph;
    auto x = graph.input("x", {1, 3});
    auto weights = graph.constant(Tensor({3, 1}, {1, 1, 1}));
    auto bias = graph.constant(Tensor({1}, 1.618f));
    auto output = graph.bias(graph.multiply(x, weights), bias);

    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 3}, {1, 2, 3})}}, cpu_backend()),
        Tensor({1, 1}, 7.618f)
    );
    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 3}, {4, 5, 6})}}, cpu_backend()),
        Tensor({1, 1}, 16.618f)
    );
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

    CountingBackend backend;
    equal(
        graph.run(output, Feeds{{"x", Tensor({1, 2}, {5, 6})}}, backend),
        Tensor({1, 2}, {4, 7})
    );
    check(backend.multiply_calls == 1, "only the live matmul should run");
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
    rejects([] { Graph graph; graph.input("bad name", {2}); }, "expected invalid name rejection");

    rejects(
        [] {
            Graph graph;
            graph.input("x", {std::numeric_limits<std::size_t>::max(), 2});
        },
        "expected element count overflow rejection"
    );
}

TINYINFER_TEST(graph, named_nodes_reject_invalid_names) {
    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.relu("x", x);
        },
        "expected duplicate name rejection"
    );
    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.relu("bad name", x);
        },
        "expected invalid name rejection"
    );
    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.relu("", x);
        },
        "expected empty name rejection"
    );
}

TINYINFER_TEST(graph, rejects_invalid_dependencies) {
    rejects([] { Graph graph; graph.multiply(0, 1); }, "expected invalid dependency rejection");

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.relu(x + 1);
        },
        "expected future dependency rejection"
    );
}

TINYINFER_TEST(graph, rejects_invalid_runs) {
    rejects(
        [] {
            Graph graph;
            const Feeds no_feeds;
            graph.run(0, no_feeds, cpu_backend());
        },
        "expected invalid output rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            const Feeds no_feeds;
            graph.run(x, no_feeds, cpu_backend());
        },
        "expected missing feed rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            graph.run(x, Feeds{{"x", Tensor({1, 2})}}, cpu_backend());
        },
        "expected feed shape rejection"
    );
}
