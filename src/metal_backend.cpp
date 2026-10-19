#include "tinyinfer/backend.hpp"

// Generate the header-only metal-cpp implementation in exactly one translation unit.
#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <QuartzCore/QuartzCore.hpp>
#include <Metal/Metal.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace tinyinfer {

namespace {

// Metal Shading Language（MSL)
const char* shader = R"metal(
    #include <metal_stdlib>

    using namespace metal;

    kernel void multiply(
        device const float* a [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* c [[buffer(2)]],        // Each thread computes the value of c at position c[pos.y][pos.x].
        constant uint3& dims [[buffer(3)]],   
        uint2 pos [[thread_position_in_grid]] // pos.x is the column index and pos.y is the row index.
    ) {
        uint row = pos.y, col = pos.x;
        if (row >= dims.x || col >= dims.z) {
            return;
        }

        float sum = 0;
        for (uint k = 0; k < dims.y; ++k) {
            sum += a[row * dims.y + k] * b[k * dims.z + col];
        }
        c[row * dims.z + col] = sum;
    }
)metal";

std::runtime_error metal_error(const char* context, NS::Error* error) {
    return std::runtime_error(
        std::string(context) + ": " + (error ? error->localizedDescription()->utf8String() : "no details")
    );
}

class MetalBackend final : public Backend {
    NS::SharedPtr<MTL::Device> device_;
    NS::SharedPtr<MTL::CommandQueue> command_queue_;
    NS::SharedPtr<MTL::ComputePipelineState> compute_pipeline_state_;

public:
    MetalBackend() {
        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

        device_ = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
        if (!device_) {
            throw std::runtime_error("no Metal device");
        }

        command_queue_ = NS::TransferPtr(device_->newCommandQueue());
        if (!command_queue_) {
            throw std::runtime_error("failed to create Metal queue");
        }

        NS::Error* error = nullptr; // Borrowed; valid until the pool drains.
        auto library = NS::TransferPtr(
            device_->newLibrary(
                NS::String::string(shader, NS::UTF8StringEncoding), 
                nullptr, 
                &error
            )
        );
        if (!library) {
            throw metal_error("shader compilation failed", error);
        }

        auto function = NS::TransferPtr(
            library->newFunction(
                NS::String::string("multiply", NS::UTF8StringEncoding)
            )
        );
        if (!function) {
            throw std::runtime_error("missing Metal multiply function");
        }
        
        compute_pipeline_state_ = NS::TransferPtr(device_->newComputePipelineState(function.get(), &error));
        if (!compute_pipeline_state_) {
            throw metal_error("pipeline creation failed", error);
        }
    }

    const char* name() const override {
        return "Metal Naive Implementation";
    }

    Tensor multiply(const Tensor& a, const Tensor& b) const override {
        if (a.shape().size() != 2 || b.shape().size() != 2 || a.shape()[1] != b.shape()[0]) {
            throw std::invalid_argument("matmul requires [M, K] and [K, N]");
        }

        const auto limit = std::numeric_limits<std::uint32_t>::max();
        const auto m = a.shape()[0], l = a.shape()[1], n = b.shape()[1];
        if (a.size() > limit || b.size() > limit || m > limit / n) {
            throw std::invalid_argument("Metal tensor exceeds uint32 indexing");
        }

        auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

        Tensor out({m, n});
        const Tensor* tensors[] = {&a, &b, &out};
        for (const Tensor* t : tensors) {
            if (t->size() * sizeof(float) > device_->maxBufferLength()) {
                throw std::invalid_argument("Metal buffer too large");
            }
        }

        auto a_buffer = NS::TransferPtr(
            device_->newBuffer(
                a.data(),
                a.size() * sizeof(float),
                MTL::ResourceStorageModeShared // Unified Memory Architecture
            )
        );
        auto b_buffer = NS::TransferPtr(
            device_->newBuffer(
                b.data(),
                b.size() * sizeof(float),
                MTL::ResourceStorageModeShared
            )
        );
        auto out_buffer = NS::TransferPtr(
            device_->newBuffer(
                out.size() * sizeof(float), 
                MTL::ResourceStorageModeShared
            )
        );

        if (!a_buffer || !b_buffer || !out_buffer) {
            throw std::runtime_error("Metal allocation failed");
        }

        // These are autoreleased, borrowed objects.
        // The pool spans encoding and execution.
        auto* command_buffer = command_queue_->commandBuffer();
        if (!command_buffer) {
            throw std::runtime_error("Metal command buffer creation failed");
        }

        auto* compute_command_encoder = command_buffer->computeCommandEncoder();
        if (!compute_command_encoder) {
            throw std::runtime_error("Metal command encoder creation failed");
        }

        compute_command_encoder->setComputePipelineState(compute_pipeline_state_.get());
        compute_command_encoder->setBuffer(a_buffer.get(), 0, 0);
        compute_command_encoder->setBuffer(b_buffer.get(), 0, 1);
        compute_command_encoder->setBuffer(out_buffer.get(), 0, 2);

        // uint3 (3 uint) occupies 16 bytes in Metal's ABI.
        const std::uint32_t dims[4] = {
            static_cast<std::uint32_t>(m),
            static_cast<std::uint32_t>(l),
            static_cast<std::uint32_t>(n),
            0                              // padding purpose
        };
        compute_command_encoder->setBytes(dims, sizeof(dims), 3);

        const auto width = std::min<NS::UInteger>(16, compute_pipeline_state_->maxTotalThreadsPerThreadgroup());
        const auto height = std::min<NS::UInteger>(16, compute_pipeline_state_->maxTotalThreadsPerThreadgroup() / width);
        compute_command_encoder->dispatchThreadgroups(
            MTL::Size((n + width - 1) / width, (m + height - 1) / height, 1), // number of Threadgroups
            MTL::Size(width, height, 1)                                       // number of threads for each Threadgroup
        );

        compute_command_encoder->endEncoding();

        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
            throw metal_error("Metal execution failed", command_buffer->error());
        }

        std::memcpy(out.data(), out_buffer->contents(), out.size() * sizeof(float));
        return out;
    }
};

} // anonymous namespace

bool metal_available() {
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());

    return static_cast<bool>(device);
}

std::unique_ptr<Backend> make_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace tinyinfer
