cmake_minimum_required(VERSION 3.21)

project(tiny-inference-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tiny_inference_engine src/tensor.cpp)
target_include_directories(tiny_inference_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny_inference_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(tiny_inference_engine_demo examples/demo.cpp)
target_link_libraries(tiny_inference_engine_demo PRIVATE tiny_inference_engine)

enable_testing()
add_executable(tiny_inference_engine_tests tests/test_main.cpp)
target_link_libraries(tiny_inference_engine_tests PRIVATE tiny_inference_engine)
add_test(NAME tiny_inference_engine_tests_cpu COMMAND tiny_inference_engine_tests)
