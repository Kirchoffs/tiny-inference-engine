cmake_minimum_required(VERSION 3.21)

project(tiny-inference-engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TINYINFER_ENABLE_METAL "Build the optional macOS Metal backend" OFF)
if(TINYINFER_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

add_library(tiny_inference_engine
    src/tensor.cpp
    src/ops.cpp
    src/graph.cpp
    src/model.cpp
)
target_include_directories(tiny_inference_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny_inference_engine PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(TINYINFER_ENABLE_METAL)
    target_sources(tiny_inference_engine PRIVATE src/metal_backend.cpp)
    include(cmake/MetalCpp.cmake)
    target_link_libraries(tiny_inference_engine PRIVATE metal_cpp::metal_cpp)
    target_compile_definitions(tiny_inference_engine PUBLIC TINYINFER_HAS_METAL=1)
endif()

file(GLOB DEMO_SOURCES CONFIGURE_DEPENDS examples/*.cpp)
foreach(demo_src ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_src} NAME_WE)
    if(demo_name STREQUAL "example_metal" AND NOT TINYINFER_ENABLE_METAL)
        continue()
    endif()

    add_executable(tiny_inference_engine_${demo_name} ${demo_src})
    target_link_libraries(tiny_inference_engine_${demo_name} PRIVATE tiny_inference_engine)
endforeach()

enable_testing()
add_executable(tiny_inference_engine_tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_ops.cpp
    tests/test_graph.cpp
    tests/test_model.cpp
    tests/test_backend.cpp
)
target_link_libraries(tiny_inference_engine_tests PRIVATE tiny_inference_engine)
target_include_directories(tiny_inference_engine_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_compile_definitions(tiny_inference_engine_tests PRIVATE
    TINYINFER_TEST_MODEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/models"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tiny_inference_engine_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME tensor_tests COMMAND tiny_inference_engine_tests tensor)
add_test(NAME ops_tests COMMAND tiny_inference_engine_tests ops)
add_test(NAME graph_tests COMMAND tiny_inference_engine_tests graph)
add_test(NAME model_tests COMMAND tiny_inference_engine_tests model)
add_test(NAME backend_tests COMMAND tiny_inference_engine_tests backend)

if(TINYINFER_ENABLE_METAL)
    target_sources(tiny_inference_engine_tests PRIVATE tests/test_metal.cpp)
    add_test(NAME metal_tests COMMAND tiny_inference_engine_tests metal)
    set_tests_properties(metal_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
