#include "tinyinfer_test_framework.hpp"

#include "tinyinfer/backend.hpp"
#include "tinyinfer/model.hpp"
#include "tinyinfer/tensor.hpp"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tinyinfer;
using namespace tinyinfer_test;

namespace {

using Feeds = std::unordered_map<std::string, Tensor>;

std::string read_fixture() {
    std::ifstream in(std::string(TINYINFER_TEST_MODEL_DIR) + "/mlp.ti", std::ios::binary);
    check(in.good(), "failed to open tests/models/mlp.ti");

    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

Model load_text(const std::string& text) {
    std::istringstream in(text);
    return Model::load(in);
}

std::string save_text(const Model& model) {
    std::ostringstream out;
    model.save(out);
    return out.str();
}

void rejects_text(const std::string& text, const char* message) {
    rejects([&] {
        std::istringstream in(text);
        (void) Model::load(in);
    }, message);
}

Feeds sample_input() {
    return Feeds{{"x", Tensor({2, 2}, {1, 2, 2, 4})}};
}

const CpuBackend& cpu_backend() {
    static const CpuBackend backend;
    return backend;
}

} // anonymous namespace

TINYINFER_TEST(model, loads_structure_and_data) {
    Model model = load_text(read_fixture());
    check(model.output() == 10, "unexpected output node");

    equal(
        model.run(sample_input(), cpu_backend()),
        Tensor({2, 2}, {0.0024726233f, 0.99752736f, 0.00012339458f, 0.99987662f})
    );
}

TINYINFER_TEST(model, preserves_node_structure_and_names) {
    const Model model = load_text(read_fixture());
    const auto text = save_text(model);

    check(text.find("input x 2 2 2") != std::string::npos, "input name and shape should be preserved");
    check(
        text.find("constant w1 2 2 3") != std::string::npos,
        "weight name and shape should be preserved"
    );
    check(text.find("matmul h1 x w1") != std::string::npos, "matmul node should be preserved");
    check(text.find("bias h2 h1 b1") != std::string::npos, "bias connections should be preserved");
    check(text.find("softmax h6 h5") != std::string::npos, "softmax connection should be preserved");
    check(text.find("output h6") != std::string::npos, "output should be preserved");
}

TINYINFER_TEST(model, round_trips_through_text) {
    const auto text = read_fixture();
    const Model model = load_text(text);
    const auto saved = save_text(model);

    check(saved == text, "saved model should match the resource file");
    equal(load_text(saved).run(sample_input(), cpu_backend()), model.run(sample_input(), cpu_backend()));
}

TINYINFER_TEST(model, saves_graph_built_in_code) {
    Graph graph;
    auto x = graph.input("x", {1, 2});
    auto w = graph.constant("w", Tensor({2, 2}, {1, 0, 0, 1}));
    auto b = graph.constant("b", Tensor({2}, 1.0f));
    auto y = graph.softmax(graph.relu(graph.bias(graph.multiply(x, w), b)));

    const Model model(std::move(graph), y);
    const auto text = save_text(model);
    check(text.find("constant w") != std::string::npos, "named weight should be saved");
    check(text.find("matmul") != std::string::npos, "topology should be saved");
    check(text.find("output") != std::string::npos, "output should be saved");

    const Model reloaded = load_text(text);
    equal(reloaded.run(Feeds{{"x", Tensor({1, 2}, {1, -1})}}, cpu_backend()),
          model.run(Feeds{{"x", Tensor({1, 2}, {1, -1})}}, cpu_backend()));
    check(save_text(reloaded) == text, "programmatic model should round-trip");
}

TINYINFER_TEST(model, generates_unique_names_for_unnamed_nodes) {
    Graph graph;
    auto x = graph.input("x", {1, 2});
    auto w = graph.constant(Tensor({2, 2}, {1, 0, 0, 1}));
    auto b = graph.constant(Tensor({2}, 1.0f));
    auto y = graph.softmax(graph.relu(graph.bias(graph.multiply(x, w), b)));

    const Model model(std::move(graph), y);
    const auto text = save_text(model);

    check(text.find("constant c1 2 2 2") != std::string::npos, "unnamed constant should be c1");
    check(text.find("constant c2 1 2") != std::string::npos, "unnamed constant should be c2");
    check(text.find("matmul n3 x c1") != std::string::npos, "unnamed matmul should be n3");
    check(text.find("bias n4 n3 c2") != std::string::npos, "unnamed bias should be n4");
    check(text.find("relu n5 n4") != std::string::npos, "unnamed relu should be n5");
    check(text.find("softmax n6 n5") != std::string::npos, "unnamed softmax should be n6");
    check(text.find("output n6") != std::string::npos, "unnamed output should be n6");

    const Feeds feeds{{"x", Tensor({1, 2}, {1, -1})}};
    const Model reloaded = load_text(text);
    equal(reloaded.run(feeds, cpu_backend()), model.run(feeds, cpu_backend()));
}

TINYINFER_TEST(model, saves_named_nodes) {
    Graph graph;
    auto x = graph.input("x", {1, 2});
    auto w = graph.constant("w", Tensor({2, 2}, {1, 0, 0, 1}));
    auto b = graph.constant("b", Tensor({2}, 1.0f));
    auto product = graph.multiply("matmul1", x, w);
    auto biased = graph.bias("bias1", product, b);
    auto activated = graph.relu("relu1", biased);
    auto y = graph.softmax("softmax1", activated);

    const Model model(std::move(graph), y);
    const auto text = save_text(model);

    check(text.find("matmul matmul1 x w") != std::string::npos, "named matmul should be saved");
    check(text.find("bias bias1 matmul1 b") != std::string::npos, "named bias should be saved");
    check(text.find("relu relu1 bias1") != std::string::npos, "named relu should be saved");
    check(text.find("softmax softmax1 relu1") != std::string::npos, "named softmax should be saved");
    check(text.find("output softmax1") != std::string::npos, "named output should be saved");

    const Model reloaded = load_text(text);
    equal(
        reloaded.run(Feeds{{"x", Tensor({1, 2}, {1, -1})}}, cpu_backend()),
        model.run(Feeds{{"x", Tensor({1, 2}, {1, -1})}}, cpu_backend())
    );
}

TINYINFER_TEST(model, supports_multiple_inputs) {
    Graph graph;
    auto a = graph.input("a", {1, 2});
    auto b = graph.input("b", {2, 1});
    auto y = graph.multiply(a, b);

    const Model model(std::move(graph), y);
    const Model reloaded = load_text(save_text(model));

    equal(
        reloaded.run(Feeds{{"a", Tensor({1, 2}, {1, 2})}, {"b", Tensor({2, 1}, {3, 4})}}, cpu_backend()),
        Tensor({1, 1}, 11.0f)
    );
}

TINYINFER_TEST(model, runs_with_cpu_backend) {
    const Model model = load_text(read_fixture());
    const CpuBackend backend;

    equal(
        model.run(sample_input(), backend),
        Tensor({2, 2}, {0.0024726233f, 0.99752736f, 0.00012339458f, 0.99987662f})
    );
}

TINYINFER_TEST(model, rejects_malformed_files) {
    rejects_text("", "expected missing header rejection");
    rejects_text("NOPE 2\nnodes 0\n", "expected bad magic rejection");
    rejects_text("TINYINFER 2\nnodes 1\ninput x 1 2\noutput x\n", "expected new version rejection");
    rejects_text("TINYINFER 1\n1\ninput x 1 2\noutput x\n", "expected missing node table rejection");
    rejects_text("TINYINFER 1\nnodes 5000\n", "expected node budget rejection");
    rejects_text("TINYINFER 1\nnodes 1\ninput x 1 2\n", "expected missing output rejection");
    rejects_text("TINYINFER 1\nnodes 2\ninput x 1 2\noutput x\n", "expected truncated node list rejection");
    rejects_text("TINYINFER 1\nnodes 1\ninput x 1 2\noutput y\n", "expected unknown output rejection");
    rejects_text(
        "TINYINFER 1\nnodes 2\ninput x 1 2\ninput x 1 2\noutput x\n",
        "expected duplicate name rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\ninput bad-name 1 2\noutput bad-name\n",
        "expected invalid name rejection"
    );
    rejects_text("TINYINFER 1\nnodes 1\nmystery n\noutput n\n", "expected unknown op rejection");
    rejects_text(
        "TINYINFER 1\nnodes 1\nmatmul n x y\noutput n\n",
        "expected unknown reference rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 2\nmatmul n x y\ninput x 1 2\noutput n\n",
        "expected forward dependency rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\nconstant c 1 2\n1\noutput c\n",
        "expected truncated data rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\nconstant c 1 2\n1 nan\noutput c\n",
        "expected non-finite rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\ninput x 1 0\noutput x\n",
        "expected zero dimension rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\ninput x 9 1 1 1 1 1 1 1 1\noutput x\n",
        "expected excessive rank rejection"
    );
    rejects_text("TINYINFER 1\nnodes -1\n", "expected invalid count rejection");
    rejects_text(
        "TINYINFER 1\nnodes 1\ninput x 0\noutput x\nextra\n",
        "expected trailing data rejection"
    );
    rejects_text(
        "TINYINFER 1\nnodes 1\ninput x 2 1048577 1\noutput x\n",
        "expected oversized dimension rejection"
    );
}

TINYINFER_TEST(model, rejects_unserializable_graphs) {
    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {1048577});
            const Model model(std::move(graph), x);
            std::ostringstream out;
            model.save(out);
        },
        "expected oversized input rejection"
    );

    rejects(
        [] {
            Graph graph;
            auto x = graph.input("x", {2});
            auto c = graph.constant(Tensor({2}, {std::numeric_limits<float>::infinity(), 0.0f}));
            auto y = graph.bias(x, c);
            const Model model(std::move(graph), y);
            std::ostringstream out;
            model.save(out);
        },
        "expected non-finite constant rejection"
    );

    rejects(
        [] {
            Graph graph;
            (void) Model(std::move(graph), 0);
        },
        "expected invalid output rejection"
    );
}
