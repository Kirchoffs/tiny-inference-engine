#pragma once

#include "tinyinfer/ops.hpp"

#include <memory>

namespace tinyinfer {

class Backend {
public:
    virtual ~Backend() = default;

    virtual const char* name() const = 0;

    // CPU defaults. A backend only needs to override the operations it
    // actually accelerates.
    virtual Tensor multiply(const Tensor& a, const Tensor& b) const {
        return tinyinfer::matmul_ikj(a, b);
    }

    virtual Tensor bias(const Tensor& x, const Tensor& b) const {
        return tinyinfer::bias(x, b);
    }

    virtual Tensor relu(const Tensor& x) const {
        return tinyinfer::relu(x);
    }

    virtual Tensor softmax(const Tensor& x) const {
        return tinyinfer::softmax(x);
    }
};

class CpuBackend final : public Backend {
public:
    const char* name() const override {
        return "CPU";
    }
};

#ifdef TINYINFER_HAS_METAL
// Throws if device/pipeline creation fails; no silent CPU fallback.
std::unique_ptr<Backend> make_metal_backend();
bool metal_available();
#endif

} // namespace tinyinfer
