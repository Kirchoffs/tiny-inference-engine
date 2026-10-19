#include "tinyinfer/ops.hpp"

#include <algorithm>
#include <cmath>

namespace tinyinfer {

Tensor matmul_ijk(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul requires [M, K] and [K, N]");
    }

    const auto m = a.shape()[0], l = a.shape()[1], n = b.shape()[1];
    
    Tensor out({m, n});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < l; ++k) {
                out.data()[i * n + j] += a.data()[i * l + k] * b.data()[k * n + j];
            }
        }
    }

    return out;
}

Tensor matmul_ikj(const Tensor& a, const Tensor& b) {
    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul requires [M, K] and [K, N]");
    }

    const auto m = a.shape()[0], l = a.shape()[1], n = b.shape()[1];

    Tensor out({m, n});
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < l; k++) {
            const auto a_element = a.data()[i * l + k]; // a[i, k]
            for (std::size_t j = 0; j < n; j++) {       // c[i, j] <- a[i, k] * b[k, j]
                out.data()[i * n + j] += a_element * b.data()[k * n + j];
            }
        }
    }

    return out;
}

Tensor matmul_blocked_ikj(const Tensor& a, const Tensor& b, std::size_t tile) {
    if (!tile) {
        throw std::invalid_argument("tile must be positive");
    }

    if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("matmul requires [M, K] and [K, N]");
    }

    const auto m = a.shape()[0], l = a.shape()[1], n = b.shape()[1];
    
    Tensor out({m, n});
    for (std::size_t is = 0; is < m;) {
        const auto ie = is + std::min(tile, m - is);
        for (std::size_t ks = 0; ks < l;) {
            const auto ke = ks + std::min(tile, l - ks);
            for (std::size_t js = 0; js < n;) {
                const auto je = js + std::min(tile, n - js);

                for (std::size_t i = is; i < ie; ++i) {
                    for (std::size_t k = ks; k < ke; ++k) {
                        const float a_element = a.data()[i * l + k];
                        for (std::size_t j = js; j < je; ++j) {
                            out.data()[i * n + j] += a_element * b.data()[k * n + j];
                        }
                    }
                }

                js = je;
            }

            ks = ke;
        }

        is = ie;
    }

    return out;
}

Tensor bias(const Tensor& x, const Tensor& b) {
    if (x.shape().size() != 2 || b.shape().size() != 1 || x.shape()[1] != b.shape()[0]) {
        throw std::invalid_argument("bias requires [M, N] and [N]");
    }

    Tensor out = x;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] += b[i % b.size()];
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
