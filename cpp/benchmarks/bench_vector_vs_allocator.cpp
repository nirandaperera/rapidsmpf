/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <rapidsmpf/error.hpp>

/**
 * @brief Benchmark to compare host allocation methods with device-to-host memory copy.
 *
 * Each benchmark creates a RMM device vector outside the timing loop, then measures
 * the combined performance of:
 * 1. Host buffer allocation (with or without initialization depending on method)
 * 2. cudaMemcpy from device to host
 *
 * All benchmarks exclude deallocation/destruction from timing to focus on
 * the allocation and memory copy overhead.
 */

void run_benchmark(benchmark::State& state, auto&& alloc, auto&& dealloc) {
    const auto size = static_cast<size_t>(state.range(0));

    auto device_mr = std::make_unique<rmm::mr::cuda_memory_resource>();
    rmm::cuda_stream_view stream = rmm::cuda_stream_default;

    // Allocate device memory
    rmm::device_buffer device_buffer(size, stream, device_mr.get());
    // Initialize device memory
    RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(device_buffer.data(), 100, size, stream));
    stream.synchronize();

    for (auto _ : state) {
        auto* ptr = alloc(size);
        RAPIDSMPF_CUDA_TRY(
            cudaMemcpy(ptr, device_buffer.data(), size, cudaMemcpyDeviceToHost)
        );

        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        dealloc(ptr, size);
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
}

// Benchmark vector creation with sized constructor
// Vector constructor will zero-initialize the memory
static void BM_Vector_Create_ZeroInit(benchmark::State& state) {
    std::unique_ptr<std::vector<uint8_t>> vec;
    run_benchmark(
        state,
        [&vec](size_t size) {
            vec = std::make_unique<std::vector<uint8_t>>(size);
            return vec->data();
        },
        [&vec](uint8_t*, size_t) { vec.reset(); }
    );
    state.SetLabel("vector create (zero-init) + cudaMemcpy");
}

// Benchmark raw allocator allocation (no initialization)
static void BM_Allocator_Allocate_NoInit(benchmark::State& state) {
    std::allocator<uint8_t> alloc;
    run_benchmark(
        state,
        [&alloc](size_t size) { return alloc.allocate(size); },
        [&alloc](uint8_t* ptr, size_t size) { alloc.deallocate(ptr, size); }
    );
    state.SetLabel("allocator allocate (no init) + cudaMemcpy");
}

// Benchmark raw allocator with explicit zero initialization
static void BM_Allocator_Allocate_ZeroInit(benchmark::State& state) {
    std::allocator<uint8_t> alloc;
    run_benchmark(
        state,
        [&alloc](size_t size) {
            auto* ptr = alloc.allocate(size);
            std::fill_n(ptr, size, uint8_t{0});
            return ptr;
        },
        [&alloc](uint8_t* ptr, size_t size) { alloc.deallocate(ptr, size); }
    );
    state.SetLabel("allocator allocate + zero-init + cudaMemcpy");
}

// Benchmark raw new[] operator (no initialization)
static void BM_New_NoInit(benchmark::State& state) {
    run_benchmark(
        state,
        [](size_t size) { return new uint8_t[size]; },
        [](uint8_t* ptr, size_t) { delete[] ptr; }
    );
    state.SetLabel("new[] (no init) + cudaMemcpy");
}

// Benchmark raw new[] with value initialization (zero-init)
static void BM_New_ZeroInit(benchmark::State& state) {
    run_benchmark(
        state,
        [](size_t size) { return new uint8_t[size](); },
        [](uint8_t* ptr, size_t) { delete[] ptr; }
    );
    state.SetLabel("new[] (zero-init) + cudaMemcpy");
}

// Benchmark calloc (allocates and zero-initializes)
// calloc is a C function that allocates memory and initializes it to zero
static void BM_Calloc_ZeroInit(benchmark::State& state) {
    run_benchmark(
        state,
        [](size_t size) { return std::calloc(size, 1); },
        [](void* ptr, size_t) { std::free(ptr); }
    );
    state.SetLabel("calloc (zero-init) + cudaMemcpy");
}

// Custom argument generator for the benchmark
// Testing various sizes from 1KB to 1GB
void CustomArguments(benchmark::internal::Benchmark* b) {
    // Small allocations (1KB - 1MB)
    // for (auto size : {
    //     1ULL << 10,      // 1 KB
    //     4ULL << 10,      // 4 KB
    //     16ULL << 10,     // 16 KB
    //     64ULL << 10,     // 64 KB
    //     256ULL << 10,    // 256 KB
    //     1ULL << 20,      // 1 MB
    // }) {
    //     b->Args({static_cast<int64_t>(size)});
    // }

    // Medium allocations (4MB - 64MB)
    for (auto size : {
             1ULL << 20,  // 1 MB
             4ULL << 20,  // 4 MB
             16ULL << 20,  // 16 MB
             32ULL << 20,  // 32 MB
             64ULL << 20,  // 64 MB
         })
    {
        b->Args({static_cast<int64_t>(size)});
    }

    // Large allocations (128MB - 1GB)
    for (auto size : {
             128ULL << 20,  // 128 MB
             256ULL << 20,  // 256 MB
             512ULL << 20,  // 512 MB
             1ULL << 30,  // 1 GB
         })
    {
        b->Args({static_cast<int64_t>(size)});
    }
}

// Register the benchmarks
BENCHMARK(BM_Vector_Create_ZeroInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Allocator_Allocate_NoInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Allocator_Allocate_ZeroInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_New_NoInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_New_ZeroInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Calloc_ZeroInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
