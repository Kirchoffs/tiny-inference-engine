#include "tinyinfer/tensor.hpp"

#include <iostream>
#include <fstream>
#include <string>

using namespace tinyinfer;

int main(int argc, char** argv) {
    try {
        (void) argc; (void) argv;

        Tensor out({2, 3}, {1, 2, 3, 4, 5, 6});
        std::cout << "row-major tensor, shape = [2, 3], at(1, 2) = " << out.at({1, 2}) << "\n";
        for (std::size_t i = 0; i < out.size(); ++i) {
            std::cout << out[i] << (i + 1 == out.size() ? '\n' : ' ');
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
