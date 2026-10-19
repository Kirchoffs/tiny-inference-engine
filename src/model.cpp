#include "tinyinfer/model.hpp"

#include "tinyinfer/backend.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tinyinfer {

namespace {

constexpr std::size_t MODEL_VERSION = 1;

constexpr std::size_t max_elements = 1048576;
constexpr std::size_t max_nodes = 4096;
constexpr std::size_t max_rank = 8;
constexpr std::size_t max_name_length = 64;

std::size_t read_size(std::istream& in, std::size_t limit) {
    std::string token;

    if (!(in >> token) || token.empty()) {
        throw std::runtime_error("missing integer");
    }

    std::size_t value = 0;
    for (char c : token) {
        if (c < '0' || c > '9' ||
            value > limit / 10 ||
            (value == limit / 10 && static_cast<std::size_t>(c - '0') > limit % 10)) {
            throw std::runtime_error("invalid or oversized integer");
        }

        value = value * 10 + static_cast<std::size_t>(c - '0');
    }

    return value;
}

bool valid_name(const std::string& name) {
    if (name.empty() || name.size() > max_name_length) {
        return false;
    }

    for (unsigned char c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '.')) {
            return false;
        }
    }

    return true;
}

std::vector<std::size_t> read_shape(std::istream& in, std::size_t& elements) {
    const auto rank = read_size(in, max_rank);
    
    std::vector<std::size_t> shape(rank);
    elements = 1;
    for (auto& axis : shape) {
        axis = read_size(in, max_elements);
        if (axis == 0 || elements > max_elements / axis) {
            throw std::runtime_error("invalid tensor dimensions");
        }
        elements *= axis;
    }

    return shape;
}

NodeId resolve(const std::unordered_map<std::string, NodeId>& nodes, const std::string& name) {
    const auto it = nodes.find(name);
    if (it == nodes.end()) {
        throw std::runtime_error("unknown node reference: " + name);
    }

    return it->second;
}

Tensor read_tensor(std::istream& in, const std::vector<std::size_t>& shape, std::size_t elements) {
    Tensor value(shape);

    for (std::size_t i = 0; i < elements; ++i) {
        std::string token;
        if (!(in >> token)) {
            throw std::runtime_error("truncated tensor data");
        }

        std::istringstream number(token);
        number.imbue(std::locale::classic());
        if (!(number >> value[i]) || 
            !std::isfinite(value[i]) || 
            number.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error("invalid tensor value");
        }
    }

    return value;
}

} // anonymous namespace

Model::Model(Graph graph, NodeId output)
    : graph_(std::move(graph)), output_(output) {
    if (output_ >= graph_.nodes_.size()) {
        throw std::invalid_argument("invalid model output");
    }
}

Model Model::load(std::istream& in) {
    std::string magic;
    if (!(in >> magic) || magic != "TINYINFER") {
        throw std::runtime_error("unsupported model header");
    }

    if (read_size(in, 9) != MODEL_VERSION) {
        throw std::runtime_error("unsupported model version");
    }

    std::string nodes_keyword;
    if (!(in >> nodes_keyword) || nodes_keyword != "nodes") {
        throw std::runtime_error("missing node table");
    }

    const auto count = read_size(in, max_nodes);

    Graph graph;
    std::unordered_map<std::string, NodeId> nodes;
    std::size_t total = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::string op;
        std::string name;
        if (!(in >> op) || !(in >> name)) {
            throw std::runtime_error("truncated node list");
        }
        if (!valid_name(name)) {
            throw std::runtime_error("invalid node name");
        }
        if (nodes.count(name)) {
            throw std::runtime_error("duplicate node name");
        }

        const NodeId id = [&]() -> NodeId {
            if (op == "input") {
                std::size_t elements = 0;
                auto shape = read_shape(in, elements);
                return graph.input(name, std::move(shape));
            }

            if (op == "constant") {
                std::size_t elements = 0;
                auto shape = read_shape(in, elements);
                if (elements > max_elements - total) {
                    throw std::runtime_error("model exceeds element budget");
                }
                total += elements;

                auto value = read_tensor(in, shape, elements);
                return graph.constant(name, std::move(value));
            }

            if (op == "matmul") {
                std::string left;
                std::string right;
                if (!(in >> left) || !(in >> right)) {
                    throw std::runtime_error("truncated matmul node");
                }
                return graph.multiply(name, resolve(nodes, left), resolve(nodes, right));
            }

            if (op == "bias") {
                std::string input;
                std::string bias;
                if (!(in >> input) || !(in >> bias)) {
                    throw std::runtime_error("truncated bias node");
                }
                return graph.bias(name, resolve(nodes, input), resolve(nodes, bias));
            }

            if (op == "relu") {
                std::string input;
                if (!(in >> input)) {
                    throw std::runtime_error("truncated relu node");
                }
                return graph.relu(name, resolve(nodes, input));
            }

            if (op == "softmax") {
                std::string input;
                if (!(in >> input)) {
                    throw std::runtime_error("truncated softmax node");
                }
                return graph.softmax(name, resolve(nodes, input));
            }

            throw std::runtime_error("unknown node op: " + op);
        }();

        nodes.emplace(name, id);
    }

    std::string output_keyword;
    std::string output_name;
    if (!(in >> output_keyword) || output_keyword != "output" || !(in >> output_name)) {
        throw std::runtime_error("missing model output");
    }

    const auto output = resolve(nodes, output_name);

    std::string extra;
    if (in >> extra) {
        throw std::runtime_error("unexpected trailing model data");
    }
    if (in.bad()) {
        throw std::runtime_error("failed to read model");
    }

    return Model(std::move(graph), output);
}

void Model::save(std::ostream& out) const {
    const auto count = graph_.nodes_.size();
    if (count > max_nodes) {
        throw std::invalid_argument("too many graph nodes");
    }
    if (output_ >= count) {
        throw std::invalid_argument("invalid model output");
    }

    std::unordered_set<std::string> used;
    std::size_t total = 0;

    // Validate first: a failure must not write a partial model.
    for (NodeId id = 0; id < count; ++id) {
        const auto& node = graph_.nodes_[id];

        if (!node.name.empty()) {
            if (!valid_name(node.name)) {
                throw std::invalid_argument("unsupported node name");
            }
            if (!used.insert(node.name).second) {
                throw std::invalid_argument("duplicate node name");
            }
        } else if (node.op == Graph::Op::Input) {
            throw std::invalid_argument("input name is empty");
        }

        if (node.op == Graph::Op::Input) {
            if (node.shape.size() > max_rank) {
                throw std::invalid_argument("input rank is too large");
            }
            if (Tensor::element_count(node.shape) > max_elements) {
                throw std::invalid_argument("input is too large");
            }
        }

        if (node.op == Graph::Op::Constant) {
            if (!node.value) {
                throw std::invalid_argument("constant has no value");
            }

            const auto& value = *node.value;
            if (value.shape().size() > max_rank) {
                throw std::invalid_argument("constant rank is too large");
            }
            if (value.size() > max_elements - total) {
                throw std::invalid_argument("too many tensor elements");
            }

            total += value.size();
            for (std::size_t i = 0; i < value.size(); ++i) {
                if (!std::isfinite(value[i])) {
                    throw std::invalid_argument("non-finite constant");
                }
            }
        }
    }

    std::vector<std::string> symbols(count);
    for (NodeId id = 0; id < count; ++id) {
        const auto& node = graph_.nodes_[id];
        if (!node.name.empty()) {
            symbols[id] = node.name;
            continue;
        }

        const std::string base =
            (node.op == Graph::Op::Constant ? "c" : "n") + std::to_string(id);
        std::string candidate = base;
        for (std::size_t suffix = 1; !used.insert(candidate).second; ++suffix) {
            candidate = base + "_" + std::to_string(suffix);
        }
        symbols[id] = candidate;
    }

    auto dependency_symbol = [&](const Graph::Node& node, std::size_t index) -> const std::string& {
        if (index >= node.inputs.size() || node.inputs[index] >= count) {
            throw std::invalid_argument("invalid graph dependency");
        }

        return symbols[node.inputs[index]];
    };

    std::ostringstream buffer;
    buffer.imbue(std::locale::classic());
    buffer << "TINYINFER " << MODEL_VERSION << '\n';
    buffer << "nodes " << count << '\n';
    buffer << std::setprecision(std::numeric_limits<float>::max_digits10);

    for (NodeId id = 0; id < count; ++id) {
        const auto& node = graph_.nodes_[id];

        switch (node.op) {
        case Graph::Op::Input: {
            buffer << "input " << symbols[id] << ' ' << node.shape.size();
            for (auto axis : node.shape) {
                buffer << ' ' << axis;
            }
            buffer << '\n';
            break;
        }
        case Graph::Op::Constant: {
            const auto& value = *node.value;
            buffer << "constant " << symbols[id] << ' ' << value.shape().size();
            for (auto axis : value.shape()) {
                buffer << ' ' << axis;
            }
            buffer << '\n';

            for (std::size_t i = 0; i < value.size(); ++i) {
                if (i != 0) {
                    buffer << ' ';
                }
                buffer << value[i];
            }
            buffer << '\n';
            break;
        }
        case Graph::Op::MatMul:
            buffer << "matmul " << symbols[id] << ' ' << dependency_symbol(node, 0) << ' ' << dependency_symbol(node, 1) << '\n';
            break;
        case Graph::Op::Bias:
            buffer << "bias " << symbols[id] << ' ' << dependency_symbol(node, 0) << ' ' << dependency_symbol(node, 1) << '\n';
            break;
        case Graph::Op::Relu:
            buffer << "relu " << symbols[id] << ' ' << dependency_symbol(node, 0) << '\n';
            break;
        case Graph::Op::Softmax:
            buffer << "softmax " << symbols[id] << ' ' << dependency_symbol(node, 0) << '\n';
            break;
        }
    }

    buffer << "output " << symbols[output_] << '\n';

    out << buffer.str();
    if (!out) {
        throw std::runtime_error("failed to write model");
    }
}

Tensor Model::run(
    const std::unordered_map<std::string, Tensor>& feeds,
    const Backend& backend
) const {
    return graph_.run(output_, feeds, backend);
}

} // namespace tinyinfer
