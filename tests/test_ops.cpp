#include "test_framework.hpp"

#include "tinyinfer/ops.hpp"
#include "tinyinfer/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <limits>

using namespace tinyinfer;
using namespace tinyinfer_test;

TINYINFER_TEST(ops, matmul_multiplies_2x3_by_3x2) {
    Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});

    equal(matmul(a, b), Tensor({2, 2}, {58, 64, 139, 154}));
}

TINYINFER_TEST(ops, matmul_with_identity_returns_input) {
    Tensor a({2, 2}, {1, 2, 3, 4});
    Tensor identity({2, 2}, {1, 0, 0, 1});

    equal(matmul(a, identity), a);
}

TINYINFER_TEST(ops, matmul_rejects_invalid_shapes) {
    rejects([] { matmul(Tensor({2, 3}), Tensor({2, 3})); }, "expected inner dimension rejection");
    rejects([] { matmul(Tensor({2}), Tensor({2, 2})); }, "expected rank rejection");
}

TINYINFER_TEST(ops, add_bias_broadcasts_over_rows) {
    Tensor x({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor bias({3}, {10, 20, 30});

    equal(add_bias(x, bias), Tensor({2, 3}, {11, 22, 33, 14, 25, 36}));
    close(x[0], 1);
}

TINYINFER_TEST(ops, add_bias_rejects_invalid_shapes) {
    rejects([] { add_bias(Tensor({2, 3}), Tensor({2})); }, "expected bias size rejection");
    rejects([] { add_bias(Tensor({2, 3}), Tensor({3, 1})); }, "expected bias rank rejection");
    rejects([] { add_bias(Tensor({3}), Tensor({3})); }, "expected input rank rejection");
}

TINYINFER_TEST(ops, relu_clamps_negative_values) {
    Tensor x({2, 3}, {-1, 0, 2, -0.5f, 3, -4});

    equal(relu(x), Tensor({2, 3}, {0, 0, 2, 0, 3, 0}));
    close(x[0], -1);
}

TINYINFER_TEST(ops, softmax_normalizes_each_row) {
    Tensor x({2, 2}, {0, 0, 1, 3});
    Tensor y = softmax(x);

    equal(y, Tensor({2, 2}, {0.5f, 0.5f, 0.11920292f, 0.88079708f}));

    for (std::size_t row = 0; row < y.shape()[0]; ++row) {
        float sum = 0;
        for (std::size_t col = 0; col < y.shape()[1]; ++col) {
            sum += y.at({row, col});
        }
        close(sum, 1.0f);
    }
}

TINYINFER_TEST(ops, softmax_is_stable_for_large_inputs) {
    Tensor y = softmax(Tensor({1, 2}, {1000, 1001}));

    close(y[0], 0.26894143f);
    close(y[1], 0.73105858f);
}

TINYINFER_TEST(ops, softmax_rejects_invalid_inputs) {
    rejects([] { softmax(Tensor({2, 3, 4})); }, "expected rank rejection");
    rejects(
        [] { softmax(Tensor({1, 2}, {0, std::numeric_limits<float>::infinity()})); },
        "expected non-finite rejection"
    );
    rejects(
        [] { softmax(Tensor({1, 2}, {0, std::numeric_limits<float>::quiet_NaN()})); },
        "expected NaN rejection"
    );
}
