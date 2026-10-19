#include "tinyinfer/ops.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace tinyinfer;

int main(int argc, char** argv) {
    try {
        (void) argc; (void) argv;

        Tensor x({2, 2}, {1, 2, 2, 4});
        Tensor w1({2, 3}, {-1, -1, -1, 1, 1, 1}), b1({3}, {1, 1, 1});
        Tensor w2({3, 2}, {2, 1, -1, -1, 2, 1}), b2({2}, {-3, -1});
        
        auto hidden = relu(bias(matmul_ijk(x, w1), b1));

        auto logits = bias(matmul_ijk(hidden, w2), b2);
        for (std::size_t i = 0; i < logits.size(); ++i) {
            std::cout << logits[i] << (i + 1 == logits.size() ? '\n' : ' ');
        }

        auto probabilities = softmax(logits);
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << probabilities[i] << (i + 1 == probabilities.size() ? '\n' : ' ');
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
