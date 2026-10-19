#include "tinyinfer_test_framework.hpp"

#include "tinyinfer/tensor.hpp"

#include <cstddef>
#include <limits>
#include <vector>

using namespace tinyinfer;
using namespace tinyinfer_test;

TINYINFER_TEST(tensor, construction_and_indexing) {
    Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});
    check(tensor.shape() == std::vector<std::size_t>{2, 3}, "shape mismatch");
    check(tensor.size() == 6, "size mismatch");
    close(tensor.at({0, 0}), 1);
    close(tensor.at({1, 2}), 6);
    close(tensor[3], 4);

    Tensor scalar({}, 3.14f);
    check(scalar.shape().empty(), "scalar shape should be empty");
    check(scalar.size() == 1, "scalar size");
    close(scalar.at({}), 3.14f);

    Tensor filled({2, 2}, 7.0f);
    equal(filled, Tensor({2, 2}, {7, 7, 7, 7}));
}

TINYINFER_TEST(tensor, copies_are_independent) {
    Tensor original({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor copy = original;
    copy[1] = 99;

    close(original[1], 2);
    close(copy[1], 99);
}

TINYINFER_TEST(tensor, rejects_invalid_indices) {
    Tensor tensor({2, 3}, {1, 2, 3, 4, 5, 6});

    rejects([&] { tensor.at({2, 1}); }, "expected out-of-range row");
    rejects([&] { tensor.at({0, 3}); }, "expected out-of-range column");
    rejects([&] { tensor.at({1}); }, "expected rank mismatch");
    rejects([&] { tensor[6]; }, "expected linear index out of range");
}

TINYINFER_TEST(tensor, rejects_invalid_shapes) {
    rejects([] { Tensor({0, 1}); }, "expected zero dimension rejection");
    rejects([] { Tensor({2, 3}, std::vector<float>{1}); }, "expected data size rejection");
    rejects(
        [] { Tensor({std::numeric_limits<std::size_t>::max(), 2}); },
        "expected element count overflow rejection"
    );
}
