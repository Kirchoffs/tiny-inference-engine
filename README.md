# Notes
## Build & Test
```text
>> mkdir build
>> cmake -S . -B build
>> cmake --build build --parallel
>> ctest --test-dir build --output-on-failure
```

```text
>> ./build/tiny_inference_engine_tests

>> ./build/tiny_inference_engine_tests tensor
>> ./build/tiny_inference_engine_tests ops
>> ./build/tiny_inference_engine_tests graph
>> ./build/tiny_inference_engine_tests model
>> ./build/tiny_inference_engine_tests backend
```

```text
>> ./build/tiny_inference_engine_example_tensor
>> ./build/tiny_inference_engine_example_ops
>> ./build/tiny_inference_engine_example_graph
>> ./build/tiny_inference_engine_example_model
>> ./build/tiny_inference_engine_example_model tests/models/mlp.ti tmp/mlp.ti
>> ./build/tiny_inference_engine_example_matmul_benchmark
>> ./build/tiny_inference_engine_example_matmul_benchmark 128 256 512
```

For meaningful benchmark numbers, use a Release build:
```text
>> cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
>> cmake --build build-release --parallel
>> ./build-release/tiny_inference_engine_example_matmul_benchmark
```

### Metal Backend
The Metal backend is optional and only available on macOS. Enabling it fetches [metal-cpp](https://github.com/apple/metal-cpp) automatically at configure time:
```text
>> cmake -S . -B build-metal -DTINYINFER_ENABLE_METAL=ON
>> cmake --build build-metal --parallel
>> ctest --test-dir build-metal --output-on-failure
>> ./build-metal/tiny_inference_engine_example_metal
>> ./build-metal/tiny_inference_engine_example_model tests/models/mlp.ti /tmp/mlp_metal.ti metal
>> ./build-metal/tiny_inference_engine_tests metal
```

The example compares the Metal result with `CpuBackend` and benchmarks both. If no Metal device is available, the metal tests are reported as skipped by CTest.

To use a local metal-cpp checkout instead of downloading it:
```text
>> cmake -S . -B build-metal -DTINYINFER_ENABLE_METAL=ON -DTINYINFER_METAL_CPP_DIR=/path/to/metal-cpp
```

`Backend` provides CPU defaults for every graph operation, and a backend only
overrides the operations it accelerates:

```c++
class Backend {
public:
    virtual const char* name() const = 0;
    virtual Tensor multiply(...) const; // defaults to matmul_ijk
    virtual Tensor bias(...) const;     // defaults to CPU
    virtual Tensor relu(...) const;     // defaults to CPU
    virtual Tensor softmax(...) const;  // defaults to CPU
};
```

`CpuBackend` overrides `multiply` with `matmul_ikj`; the Metal backend overrides
only `multiply` and inherits the CPU implementations of the remaining operations.

Graph and model execution take a backend explicitly:

```c++
CpuBackend cpu;
auto result = graph.run(output, feeds, cpu);
auto probabilities = model.run(feeds, cpu);

#ifdef TINYINFER_HAS_METAL
auto metal = make_metal_backend();
auto gpu_probabilities = model.run(feeds, *metal);
#endif
```

## Design
### Architecture
1\. Tensor  
`Tensor` is the lowest-level data abstraction. It owns a contiguous row-major `float` buffer together with a shape, supports scalar tensors, multidimensional indexing, and validates bounds, zero dimensions, and size overflow. All higher layers exchange data through `Tensor`.

2\. Operators  
The operator layer consists of stateless functions over tensors:
- `matmul_ijk`, `matmul_ikj`, `matmul_blocked_ikj`
- `bias`
- `relu`
- `softmax`

There is no polymorphic `Operator` base class or runtime operator object. An operator is represented primarily by an `Op` enum inside a graph node, and the actual computation is implemented by these free functions or by a backend method. This keeps the operator implementations simple, testable, and independent of graph structure.

3\. Graph and Node  
`Graph` is a directed acyclic graph of computation nodes. Each `Node` stores:
- its `Op` kind;
- an optional symbolic name;
- a list of input/dependency node ids;
- a shape for `Input` nodes;
- an optional tensor value for `Constant` nodes.

A `NodeId` is currently the index of the node in the graph's node vector. Nodes are created through builder methods such as `input`, `constant`, `multiply`, `bias`, `relu`, and `softmax`. Each builder appends a new node and returns its id.

A dependency may only refer to a node that already exists. Consequently, __insertion order is already a valid topological order__; no separate DFS/Kahn topological-sort pass is required. The cost is that callers and the model loader must respect the same construction order. Forward references and cycles are therefore rejected at construction time.

`Graph::run` evaluates nodes in vector order. To avoid unnecessary work, it first walks backward from the requested output and marks only the nodes actually needed, then evaluates that subset. Unused nodes and missing feeds for unused branches are ignored.

4\. Backend  
`Backend` abstracts graph execution:
```text
Backend
├── CpuBackend
└── MetalBackend
```

The base `Backend` provides CPU defaults for `multiply`, `bias`, `relu`, and `softmax`. `CpuBackend` overrides `multiply` with `matmul_ikj`; the Metal backend overrides only `multiply` and inherits the CPU implementations of the other operations. This lets a backend accelerate just the operators it supports without implementing the full set.

One clarification: the backend is **not owned by Graph**. Instead, it is passed into `Graph::run(output, feeds, backend)` and `Model::run(feeds, backend)`. Graph depends on the abstract `Backend` interface, not on a concrete CPU or Metal implementation.

5\. Model  
`Model` adds persistence on top of `Graph`:
- it owns a `Graph` and identifies the output node;
- `save` serializes both graph topology and tensor data into a self-describing text resource;
- `load` reconstructs the graph by replaying the builder calls in file order.

The text format contains the version, node count, node records (`input`, `constant`, `matmul`, `bias`, `relu`, `softmax`), inline constant values, and an explicit output symbol. Loading validates names, references, shapes, element budgets, finite values, and trailing data.

For unnamed `Constant` and computed nodes, `save` generates deterministic unique symbols such as `c3` or `n5`, handling collisions with suffixes. `Input` names must remain explicit because they are also the keys used in the feed map.

### Model Files
A `.ti` file now contains both the graph topology and its tensor data:
```text
TINYINFER 1
nodes 11
input x 2 2 2
constant w1 2 2 3
1 -1 -1 -1 -1 1
...
matmul h1 x w1
bias h2 h1 b1
relu h3 h2
...
softmax h6 h5
output h6
```

Nodes are declared in topological order; names are unique symbols and may be referenced by later nodes. Constants store their values inline.  
The C++ API is `Model::load(std::istream&)`, `model.save(std::ostream&)` and `model.run(feeds, backend)`.

## Miscellaneous
### Time
#### std::chrono::steady_clock::duration
```c++
double milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}
```

```c++
auto start = std::chrono::steady_clock::now();
process();
auto end = std::chrono::steady_clock::now();

auto elapsed_steady_clock_duration = end - start; // std::chrono::steady_clock::duration or std::chrono::duration<long long, std::nano>
auto elapsed_steady_duration = std::chrono::duration<double, std::milli>(
    elapsed_steady_clock_duration
);                                                // std::chrono::duration<doube, std::milli>
auto elased = elapsed_steady_duration.count();    // double
```

### Raw String Literal
Introduced in C++11

```c++
const char* str = R"delimiter("Hello World!")delimiter";
```

### CMake
```cmake
add_executable(tiny_inference_engine_tests
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_ops.cpp
    tests/test_graph.cpp
    tests/test_model.cpp
    tests/test_backend.cpp
)
# [test_main.cpp, test_tensor.cpp, test_ops.cpp, test_graph.cpp, test_model.cpp, test_backend.cpp]

if(TINYINFER_ENABLE_METAL)
target_sources(tiny_inference_engine_tests PRIVATE tests/test_metal.cpp)
# [test_main.cpp, test_tensor.cpp, test_ops.cpp, test_graph.cpp, test_model.cpp, test_backend.cpp, test_metal.cpp]
endif()
```

### Metal
#### Object Ownership
In `metal-cpp`, the objects you get fall into two categories, and they are handled completely differently:

1. Objects you "own" (You must release them)
When you call a method whose name starts with `new`, `alloc`, `copy`, `mutableCopy`, or `Create`, the returned object has a retain count of 1, and you own it.  
Examples include `device->newBuffer(...)` and `MTL::TextureDescriptor::alloc()->init()`. You are responsible for releasing these objects; otherwise, you'll get a memory leak.

2. "Autoreleased" objects (Not your responsibility, hands off)
Objects returned by other methods (such as `*CommandEncoder()` or `::descriptor()`) are managed by an autorelease pool. You don't need to (and shouldn't) manually release them—the system will clean them up automatically when the pool is drained.

#### NS::TransferPtr
Its purpose is to transfer a raw pointer you already own to an `NS::SharedPtr` smart pointer for management, without adding an extra reference count.

The key here is "without incrementing". Since you already hold an implicit retain count (+1), TransferPtr just transfers that ownership to the smart pointer without an extra retain. When the smart pointer is destroyed, it issues a single release, perfectly balancing out the initial retain.

Therefore, you should use `TransferPtr` only when you obtain an object from `new`, `alloc`, `copy`, or `Create`.

#### Metal Compute Architecture
- Device -> CommandQueue -> CommandBuffer -> ComputeCommandEncoder
- Device -> Library -> Function -> ComputePipelineState

The ComputeCommandEncoder encodes GPU commands using the ComputePipelineState, and the CommandBuffer submits them for execution.
