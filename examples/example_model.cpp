#include "tinyinfer/backend.hpp"
#include "tinyinfer/model.hpp"
#include "tinyinfer/tensor.hpp"

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

using namespace tinyinfer;

namespace {

bool same(const Tensor& a, const Tensor& b, float tolerance = 1e-6f) {
    if (a.shape() != b.shape()) {
        return false;
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance * (1.0f + std::abs(b[i]))) {
            return false;
        }
    }

    return true;
}

} // anonymous namespace

int main(int argc, char** argv) {
    try {
        const std::string input_path = argc > 1 ? argv[1] : "tests/models/mlp.ti";
        const std::string output_path = argc > 2 ? argv[2] : "tmp/mlp_roundtrip.ti";

        std::ifstream input(input_path);
        if (!input) {
            throw std::runtime_error("cannot open model file: " + input_path);
        }

        const Model model = Model::load(input);
        std::cout << "loaded " << input_path << '\n';

        const std::string backend_name = argc > 3 ? argv[3] : "cpu";
        std::unique_ptr<Backend> backend;
        if (backend_name == "cpu") {
            backend = std::make_unique<CpuBackend>();
        } else if (backend_name == "metal") {
#ifdef TINYINFER_HAS_METAL
            if (!metal_available()) {
                throw std::runtime_error("Metal device is not available");
            }
            backend = make_metal_backend();
#else
            throw std::runtime_error(
                "Metal backend is not enabled; configure with -DTINYINFER_ENABLE_METAL=ON"
            );
#endif
        } else {
            throw std::runtime_error("unknown backend: " + backend_name);
        }
        std::cout << "backend: " << backend->name() << '\n';

        const std::unordered_map<std::string, Tensor> feeds{
            {"x", Tensor({2, 2}, {1, 2, 2, 4})}
        };
        const auto probabilities = model.run(feeds, *backend);

        std::cout << "probabilities:";
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            std::cout << ' ' << probabilities[i];
        }
        std::cout << '\n';

        std::ofstream output(output_path);
        if (!output) {
            throw std::runtime_error("cannot write model file: " + output_path);
        }
        model.save(output);
        output.close();
        if (!output) {
            throw std::runtime_error("failed to write model file: " + output_path);
        }
        std::cout << "saved model: " << output_path << '\n';

        std::ifstream saved(output_path);
        if (!saved) {
            throw std::runtime_error("cannot read back model file: " + output_path);
        }
        const Model reloaded = Model::load(saved);
        if (!same(probabilities, reloaded.run(feeds, *backend))) {
            throw std::runtime_error("reloaded model produced different output");
        }

        std::cout << "round-trip check: ok\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
