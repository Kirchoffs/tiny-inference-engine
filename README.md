# Notes
## Build & Test
```
>> mkdir build
>> cmake -S . -B build
>> cmake --build build --parallel
>> ctest --test-dir build --output-on-failure
```

```
>> ./build/tiny_inference_engine_tests
>> ./build/tiny_inference_engine_tests ops
```

```
>> ./build/tiny_inference_engine_demo_01
>> ./build/tiny_inference_engine_demo_02
```
