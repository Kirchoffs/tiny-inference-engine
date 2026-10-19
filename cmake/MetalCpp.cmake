include_guard(GLOBAL)

include(FetchContent)

set(
    TINYINFER_METAL_CPP_DIR
    ""
    CACHE PATH
    "Path to a metal-cpp checkout; fetched from GitHub when empty"
)

if(TINYINFER_METAL_CPP_DIR)
    set(_tinyinfer_metal_cpp_include "${TINYINFER_METAL_CPP_DIR}")
else()
    FetchContent_Declare(
        metal_cpp
        GIT_REPOSITORY https://github.com/apple/metal-cpp.git
        GIT_TAG release/metal-cpp_macOS26.4_iOS26.4
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(metal_cpp)
    set(_tinyinfer_metal_cpp_include "${metal_cpp_SOURCE_DIR}")
endif()

if(NOT EXISTS "${_tinyinfer_metal_cpp_include}/Metal/Metal.hpp")
    message(FATAL_ERROR "metal-cpp headers not found in ${_tinyinfer_metal_cpp_include}")
endif()

add_library(metal_cpp INTERFACE)
add_library(metal_cpp::metal_cpp ALIAS metal_cpp)

target_include_directories(metal_cpp SYSTEM INTERFACE "${_tinyinfer_metal_cpp_include}")
target_compile_features(metal_cpp INTERFACE cxx_std_17)
target_link_libraries(
    metal_cpp
    INTERFACE
        "-framework Foundation"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework QuartzCore"
)
