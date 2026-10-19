#pragma once

#include "tinyinfer/graph.hpp"

#include <istream>
#include <ostream>
#include <unordered_map>

namespace tinyinfer {

// A model is a graph together with the node whose value is the output.
//
// Text format (TINYINFER 1), where every name is a unique symbol and every
// reference must point to an earlier node:
//
//   TINYINFER 1
//   nodes <count>
//   input <name> <rank> <dim>...
//   constant <name> <rank> <dim>...
//   <value>...
//   matmul <name> <left> <right>
//   bias <name> <input> <bias>
//   relu <name> <input>
//   softmax <name> <input>
//   output <name>
//
// Constant values are stored inline, directly after their shape. This keeps
// topology and tensor data in one self-contained resource file.
class Model {
public:
    Model(Graph graph, NodeId output);

    static Model load(std::istream& in);
    void save(std::ostream& out) const;

    Tensor run(
        const std::unordered_map<std::string, Tensor>& feeds,
        const Backend& backend
    ) const;

    const Graph& graph() const {
        return graph_;
    }

    NodeId output() const {
        return output_;
    }

private:
    Graph graph_;
    NodeId output_;
};

} // namespace tinyinfer
