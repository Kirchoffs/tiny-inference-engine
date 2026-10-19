#pragma once

#include "tinyinfer/ops.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer {

class Backend;
class Model;

using NodeId = std::size_t;

class Graph {
public:
    friend class Model;

    NodeId input(std::string name, std::vector<std::size_t> shape);
    NodeId constant(Tensor value);
    NodeId constant(std::string name, Tensor value);
    NodeId multiply(NodeId a, NodeId b);
    NodeId multiply(std::string name, NodeId a, NodeId b);
    NodeId bias(NodeId x, NodeId b);
    NodeId bias(std::string name, NodeId x, NodeId b);
    NodeId relu(NodeId x);
    NodeId relu(std::string name, NodeId x);
    NodeId softmax(NodeId x);
    NodeId softmax(std::string name, NodeId x);

    Tensor run(
        NodeId output,
        const std::unordered_map<std::string, Tensor>& feeds,
        const Backend& backend
    ) const;

private:
    enum class Op { Input, Constant, MatMul, Bias, Relu, Softmax };

    struct Node {
        Op op;
        std::string name;
        std::vector<NodeId> inputs;
        std::vector<std::size_t> shape;
        std::optional<Tensor> value;
    };

    NodeId append(Op op, std::vector<NodeId> inputs);
    NodeId append_named(Op op, std::string name, std::vector<NodeId> inputs);
    void require_unique_name(const std::string& name) const;

    std::vector<Node> nodes_;
};

} // namespace tinyinfer
