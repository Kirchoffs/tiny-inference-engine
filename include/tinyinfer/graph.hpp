#pragma once

#include "tinyinfer/ops.hpp"

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>

namespace tinyinfer {

using NodeId = std::size_t;
using MatMulKernel = std::function<Tensor(const Tensor&, const Tensor&)>;

class Graph {
public:
    NodeId input(std::string name, std::vector<std::size_t> shape);
    NodeId constant(Tensor value);
    NodeId multiply(NodeId a, NodeId b);
    NodeId bias(NodeId x, NodeId b);
    NodeId activation(NodeId x);
    NodeId probabilities(NodeId x);
    Tensor run(
        NodeId output,
        const std::unordered_map<std::string, Tensor>& feeds,
        MatMulKernel kernel = matmul
    ) const;

private:
    enum class Op { Input, Constant, MatMul, Bias, Relu, Softmax };

    struct Node {
        std::string name;
        Op op;
        std::vector<NodeId> inputs;
        std::vector<std::size_t> shape;
        std::optional<Tensor> value;
    };

    NodeId append(Op op, std::vector<NodeId> inputs);

    std::vector<Node> nodes_;
};

} // namespace tinyinfer
