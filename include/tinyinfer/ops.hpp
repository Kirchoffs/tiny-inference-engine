#pragma once

#include "tinyinfer/tensor.hpp"

namespace tinyinfer {
    Tensor matmul_ijk(const Tensor& a, const Tensor& b);
    Tensor matmul_ikj(const Tensor& a, const Tensor& b);
    Tensor matmul_blocked_ikj(const Tensor& a, const Tensor& b, std::size_t tile = 32);

    Tensor bias(const Tensor& x, const Tensor& b);
    Tensor relu(const Tensor& x);
    Tensor softmax(const Tensor& x);
} // namespace tinyinfer
