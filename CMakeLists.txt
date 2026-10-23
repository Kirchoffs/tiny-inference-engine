cmake_minimum_required(VERSION 3.21)

project(tiny-inference-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tiny_inference_engine
    src/tensor.cpp
    src/ops.cpp
    src/graph.cpp
)
target_include_directories(tiny_inference_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny_inference_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

file(GLOB DEMO_SOURCES CONFIGURE_DEPENDS examples/demo_*.cpp)
foreach(demo_src ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_src} NAME_WE)
    add_executable(tiny_inference_engine_${demo_name} ${demo_src})
    target_link_libraries(tiny_inference_engine_${demo_name} PRIVATE tiny_inference_engine)
endforeach()

enable_testing()
add_executable(tiny_inference_engine_tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_ops.cpp
    tests/test_graph.cpp
)
target_link_libraries(tiny_inference_engine_tests PRIVATE tiny_inference_engine)
target_include_directories(tiny_inference_engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny_inference_engine_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME tensor_tests COMMAND tiny_inference_engine_tests tensor)
add_test(NAME ops_tests COMMAND tiny_inference_engine_tests ops)
add_test(NAME graph_tests COMMAND tiny_inference_engine_tests graph)
