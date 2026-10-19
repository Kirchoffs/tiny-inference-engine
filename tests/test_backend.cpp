#include "tinyinfer_test_framework.hpp"

#include "tinyinfer/backend.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/tensor.hpp"

#include <string>

using namespace tinyinfer;
using namespace tinyinfer_test;

TINYINFER_TEST(backend, cpu_backend_reports_name) {
    const CpuBackend backend;
    check(std::string(backend.name()) == "CPU", "unexpected backend name");
}

TINYINFER_TEST(backend, cpu_backend_multiplies) {
    const CpuBackend backend;
    const Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});

    equal(backend.multiply(a, b), matmul_ikj(a, b));
}

TINYINFER_TEST(backend, cpu_backend_rejects_invalid_shapes) {
    const CpuBackend backend;

    rejects(
        [&] { backend.multiply(Tensor({2, 3}), Tensor({2, 3})); },
        "expected inner dimension rejection"
    );
    rejects(
        [&] { backend.multiply(Tensor({2}), Tensor({2, 2})); },
        "expected rank rejection"
    );
}
