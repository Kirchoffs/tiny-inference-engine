#include "tinyinfer/graph.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace tinyinfer;

int main(int argc, char** argv) {
    try {
        (void) argc; (void) argv;
        
        Graph graph;

        auto x = graph.input("x", {2, 2});
        auto w1 = graph.constant(Tensor({2, 3},{-1, -1, -1, 1, 1, 1}));
        auto b1 = graph.constant(Tensor({3}, {1, 2, 3}));
        auto y1 = graph.activation(graph.bias(graph.multiply(x, w1), b1));
        auto w2 = graph.constant(Tensor({3, 2}, {1, -1, 1, -1, 1, -1}));
        auto b2 = graph.constant(Tensor({2}, {1, 1}));
        auto y2 = graph.probabilities(graph.bias(graph.multiply(y1, w2), b2));
        auto o = graph.run(y2, {{"x", Tensor({2, 2}, {1, 2, 1, 2})}});
        
        for (std::size_t i = 0; i < o.size(); ++i) {
            std::cout << o[i] << (i + 1 == o.size() ? '\n' : ' ');
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
