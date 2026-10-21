#include "tinyinfer/ops.hpp"
#include <algorithm>
#include <cmath>

namespace tinyinfer {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul requires [M, K] and [K, N]");
    }

    const auto m = a.shape()[0], k = a.shape()[1], n = b.shape()[1];
    Tensor out({m, n});

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t p = 0; p < k; ++p) {
                out.data()[i * n + j] += a.data()[i * k + p] * b.data()[p * n + j];
            }
        }
    }

    return out;
}

Tensor add_bias(const Tensor& x, const Tensor& bias) {
    if (x.shape().size() != 2 || bias.shape().size() != 1 || x.shape()[1] != bias.shape()[0]) {
        throw std::invalid_argument("add_bias requires [M, N] and [N]");
    }

    Tensor out = x;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] += bias[i % bias.size()];
    }
    
    return out;
}

Tensor relu(const Tensor& x) {
    Tensor out = x;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = std::max(0.0f, out[i]);
    }

    return out;
}

Tensor softmax(const Tensor& x) {
    if (x.shape().size() != 2) {
        throw std::invalid_argument("softmax requires [batch, classes]");
    }

    Tensor out(x.shape());
    auto n = x.shape()[1];
    for (std::size_t row = 0; row < x.shape()[0]; ++row) {
        float maximum = x[row * n];
        for (std::size_t j = 0; j < n; ++j) {
            if (!std::isfinite(x[row * n + j])) {
                throw std::invalid_argument("softmax requires finite inputs");
            }

            maximum = std::max(maximum, x[row * n + j]);
        }

        double sum = 0;
        for (std::size_t j = 0; j < n; ++j) {
            out[row * n + j] = std::exp(x[row * n + j] - maximum);
            sum += out[row*n+j];
        }

        for (std::size_t j = 0; j < n; ++j) {
            out[row * n + j] = static_cast<float>(out[row * n + j] / sum);
        }
    }

    return out;
}

} // namespace tinyinfer
