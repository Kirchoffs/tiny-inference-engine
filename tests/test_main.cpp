#include "tinyinfer_test_framework.hpp"

#include <string>

int main(int argc, char** argv) {
    const std::string suite = argc > 1 ? argv[1] : "";
    return tinyinfer_test::run_all(suite);
}
