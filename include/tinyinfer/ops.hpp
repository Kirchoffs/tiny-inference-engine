#pragma once

#include "tinyinfer/tensor.hpp"

namespace tinyinfer {
    Tensor matmul(const Tensor& a, const Tensor& b);
    Tensor add_bias(const Tensor& x, const Tensor& bias);
    Tensor relu(const Tensor& x);
    Tensor softmax(const Tensor& x);
} // namespace tinyinfer
