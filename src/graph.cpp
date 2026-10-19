#include "tinyinfer/graph.hpp"

#include "tinyinfer/backend.hpp"

#include <utility>

namespace tinyinfer {

namespace {

constexpr std::size_t max_name_length = 64;

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

} // anonymous namespace

void Graph::require_unique_name(const std::string& name) const {
    for (const auto& node : nodes_) {
        if (node.name == name) {
            throw std::invalid_argument("duplicate node name");
        }
    }
}

NodeId Graph::input(std::string name, std::vector<std::size_t> shape) {
    if (!valid_name(name)) {
        throw std::invalid_argument("invalid input name");
    }

    require_unique_name(name);
    Tensor::element_count(shape);
    nodes_.push_back({Op::Input, std::move(name), {}, std::move(shape), std::nullopt});

    return nodes_.size() - 1;
}

NodeId Graph::constant(Tensor value) {
    nodes_.push_back({Op::Constant, {}, {}, {}, std::move(value)});
    return nodes_.size() - 1;
}

NodeId Graph::constant(std::string name, Tensor value) {
    if (!valid_name(name)) {
        throw std::invalid_argument("invalid constant name");
    }

    require_unique_name(name);
    nodes_.push_back({Op::Constant, std::move(name), {}, {}, std::move(value)});

    return nodes_.size() - 1;
}

NodeId Graph::multiply(NodeId a, NodeId b) { return append(Op::MatMul, {a, b}); }

NodeId Graph::multiply(std::string name, NodeId a, NodeId b) {
    return append_named(Op::MatMul, std::move(name), {a, b});
}

NodeId Graph::bias(NodeId x, NodeId b) { return append(Op::Bias, {x, b}); }

NodeId Graph::bias(std::string name, NodeId x, NodeId b) {
    return append_named(Op::Bias, std::move(name), {x, b});
}

NodeId Graph::relu(NodeId x) { return append(Op::Relu, {x}); }

NodeId Graph::relu(std::string name, NodeId x) {
    return append_named(Op::Relu, std::move(name), {x});
}

NodeId Graph::softmax(NodeId x) { return append(Op::Softmax, {x}); }

NodeId Graph::softmax(std::string name, NodeId x) {
    return append_named(Op::Softmax, std::move(name), {x});
}

NodeId Graph::append(Op op, std::vector<NodeId> inputs) {
    // Edges can only point to existing nodes: insertion order is topological.
    for (auto id : inputs) {
        if (id >= nodes_.size()) {
            throw std::out_of_range("invalid graph dependency");
        }
    }
    nodes_.push_back({op, {}, std::move(inputs), {}, std::nullopt});

    return nodes_.size() - 1;
}

NodeId Graph::append_named(Op op, std::string name, std::vector<NodeId> inputs) {
    if (!valid_name(name)) {
        throw std::invalid_argument("invalid node name");
    }
    require_unique_name(name);

    const NodeId id = append(op, std::move(inputs));
    nodes_[id].name = std::move(name);
    return id;
}

Tensor Graph::run(
    NodeId output,
    const std::unordered_map<std::string, Tensor>& feeds,
    const Backend& backend
) const {
    if (output >= nodes_.size()) {
        throw std::out_of_range("invalid graph output");
    }

    std::vector<bool> needed(nodes_.size(), false);
    needed[output] = true;
    for (std::size_t i = output + 1; i-- > 0;) {
        if (needed[i]) {
            for (auto dep : nodes_[i].inputs) {
                needed[dep] = true;
            }
        }
    }

    std::vector<std::optional<Tensor>> values(nodes_.size());
    for (std::size_t i = 0; i <= output; ++i) {
        if (!needed[i]) {
            continue;
        }

        const auto& node = nodes_[i];
        auto arg = [&](std::size_t j) -> const Tensor& { 
            return values[node.inputs[j]].value();
        };

        switch (node.op) {
        case Op::Input: {
            auto it = feeds.find(node.name);
            if (it == feeds.end()) {
                throw std::invalid_argument("missing graph input: " + node.name);
            }
            if (it->second.shape() != node.shape) {
                throw std::invalid_argument("graph input shape mismatch");
            }

            values[i] = it->second;
            break;
        }
        case Op::Constant: values[i] = node.value; break;
        case Op::MatMul: values[i] = backend.multiply(arg(0), arg(1)); break;
        case Op::Bias: values[i] = backend.bias(arg(0), arg(1)); break;
        case Op::Relu: values[i] = backend.relu(arg(0)); break;
        case Op::Softmax: values[i] = backend.softmax(arg(0)); break;
        }
    }
    return std::move(values[output].value());
}

} // namespace tinyinfer
