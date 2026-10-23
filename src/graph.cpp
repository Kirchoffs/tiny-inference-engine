#include "tinyinfer/graph.hpp"

namespace tinyinfer {

NodeId Graph::input(std::string name, std::vector<std::size_t> shape) {
    if (name.empty()) {
        throw std::invalid_argument("input name is empty");
    }

    Tensor::element_count(shape);

    for (const auto& node : nodes_) {
        if (node.op == Op::Input && node.name == name) {
            throw std::invalid_argument("duplicate input name");
        }
    }
    nodes_.push_back({std::move(name), Op::Input, {}, std::move(shape), std::nullopt});

    return nodes_.size() - 1;
}

NodeId Graph::constant(Tensor value) {
    nodes_.push_back({{}, Op::Constant, {}, {}, std::move(value)});
    return nodes_.size() - 1;
}

NodeId Graph::append(Op op, std::vector<NodeId> inputs) {
    // Edges can only point to existing nodes: insertion order is topological.
    for (auto id : inputs) {
        if (id >= nodes_.size()) {
            throw std::out_of_range("invalid graph dependency");
        }
    }
    nodes_.push_back({{}, op, std::move(inputs), {}, std::nullopt});

    return nodes_.size() - 1;
}

NodeId Graph::multiply(NodeId a, NodeId b) { return append(Op::MatMul, {a, b}); }

NodeId Graph::bias(NodeId x, NodeId b) { return append(Op::Bias, {x, b}); }

NodeId Graph::activation(NodeId x) { return append(Op::Relu, {x}); }

NodeId Graph::probabilities(NodeId x) { return append(Op::Softmax, {x}); }

Tensor Graph::run(NodeId output, const std::unordered_map<std::string, Tensor>& feeds, MatMulKernel kernel) const {
    if (output >= nodes_.size()) {
        throw std::out_of_range("invalid graph output");
    }

    if (!kernel) {
        throw std::invalid_argument("empty matmul kernel");
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
        case Op::MatMul: values[i] = kernel(arg(0), arg(1)); break;
        case Op::Bias: values[i] = add_bias(arg(0), arg(1)); break;
        case Op::Relu: values[i] = relu(arg(0)); break;
        case Op::Softmax: values[i] = softmax(arg(0)); break;
        }
    }
    return std::move(values[output].value());
}

} // namespace tinyinfer
