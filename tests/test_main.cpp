#include "tinyinfer/tensor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <random>

using namespace tinyinfer;

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F>
void rejects(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception&) {
        caught = true;
    }

    check(caught, "expected exception");
}

void close(float a, float b, float tolerance = 1e-5f) {
    check(
        std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance * (1 + std::abs(b)),
        "numerical mismatch"
    );
}

void equal(const Tensor& a, const Tensor& b, float tolerance = 1e-5f) {
    check(a.shape() == b.shape(), "shape mismatch");

    for (std::size_t i = 0; i < a.size(); ++i) {
        close(a[i], b[i], tolerance);
    }
}

int main() {
    try {
        Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});
        close(tensor.at({1, 2}), 6);
        close(tensor[3], 4);

        Tensor tensor_copy = tensor;
        tensor_copy[1] = 99;
        close(tensor[1], 2);

        Tensor scalar({}, 3.14f);
        check(scalar.size() == 1, "scalar size");
        close(scalar.at({}), 3.14f);

        rejects([&]{ tensor.at({2, 1}); });
        rejects([&]{ tensor.at({1}); });
        rejects([&]{ tensor[6]; });

        rejects([]{ Tensor({0, 1}); }); 
        rejects([]{ Tensor({2, 3}, std::vector<float>{1}); });
        rejects([]{ Tensor({std::numeric_limits<std::size_t>::max(), 2}); });
        
        std::cout << "All CPU tests passed.\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
