#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tinyinfer {

class Tensor {
public:
    explicit Tensor(std::vector<std::size_t> shape, float value = 0.0f)
        : shape_(std::move(shape)), data_(element_count(shape_), value) {}

    Tensor(std::vector<std::size_t> shape, std::vector<float> values)
        : shape_(std::move(shape)), data_(std::move(values)) {
        if (data_.size() != element_count(shape_)) {
            throw std::invalid_argument("tensor data size does not match shape");
        }
    }

    static std::size_t element_count(const std::vector<std::size_t>& shape) {
        std::size_t n = 1;
        for (auto d : shape) {
            if (d == 0) {
                throw std::invalid_argument("zero dimensions are unsupported");
            }

            if (n > std::numeric_limits<std::size_t>::max() / d) {
                throw std::overflow_error("tensor element count overflow");
            }

            n *= d;
        }

        if (n > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("tensor byte count overflow");
        }

        return n;
    }

    const std::vector<std::size_t>& shape() const { 
        return shape_;
    }

    std::size_t size() const {
        return data_.size();
    }

    const float* data() const {
        return data_.data();
    }

    float* data() {
        return data_.data();
    }

    float& operator[](std::size_t i) {
        return data_.at(i);
    }

    const float& operator[](std::size_t i) const {
        return data_.at(i);
    }

    float& at(const std::vector<std::size_t>& indices) {
        return data_.at(offset(indices));
    }

    const float& at(const std::vector<std::size_t>& indices) const {
        return data_.at(offset(indices));
    }

private:
    std::size_t offset(const std::vector<std::size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("index rank mismatch");
        }

        std::size_t result = 0;
        for (std::size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("tensor index out of range");
            }

            result = result * shape_[i] + indices[i];
        }

        return result;
    }

    std::vector<std::size_t> shape_;
    std::vector<float> data_;
};

} // namespace tinyinfer
