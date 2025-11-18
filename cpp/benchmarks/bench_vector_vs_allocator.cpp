/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

/**
 * @brief Benchmark to compare std::vector<uint8_t> creation vs std::allocator allocation.
 *
 * The key difference: std::vector constructor zero-initializes the memory region,
 * while std::allocator::allocate does not initialize memory. This benchmark
 * measures the performance impact of that zero-initialization.
 *
 * All benchmarks exclude deallocation/destruction from timing to focus on
 * the allocation and initialization overhead.
 */

// Benchmark vector creation with sized constructor
// Vector constructor will zero-initialize the memory
static void BM_Vector_Create_ZeroInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        auto vec = std::make_unique<std::vector<uint8_t>>(size);
        auto* ptr = vec->data();
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        vec.reset();  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("vector create (zero-init)");
}

// Benchmark vector creation with reserve (no initialization)
// This only allocates capacity without constructing elements
static void BM_Vector_Reserve_NoInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        auto vec = std::make_unique<std::vector<uint8_t>>();
        vec->reserve(size);
        auto* ptr = vec->data();
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        vec.reset();  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("vector reserve (no init)");
}

// Benchmark vector resize (will zero-initialize)
// resize() allocates and value-initializes elements
static void BM_Vector_Resize_ZeroInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        auto vec = std::make_unique<std::vector<uint8_t>>();
        vec->resize(size);
        auto* ptr = vec->data();
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        vec.reset();  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("vector resize (zero-init)");
}

// Benchmark raw allocator allocation (no initialization)
static void BM_Allocator_Allocate_NoInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));
    std::allocator<uint8_t> alloc;

    for (auto _ : state) {
        uint8_t* ptr = alloc.allocate(size);
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        alloc.deallocate(ptr, size);  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("allocator allocate (no init)");
}

// Benchmark raw allocator with explicit zero initialization
static void BM_Allocator_Allocate_ZeroInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));
    std::allocator<uint8_t> alloc;

    for (auto _ : state) {
        uint8_t* ptr = alloc.allocate(size);
        // Explicitly zero-initialize like vector does
        std::fill_n(ptr, size, uint8_t{0});
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        alloc.deallocate(ptr, size);  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("allocator allocate + zero-init");
}

// Benchmark raw new[] operator (no initialization)
static void BM_New_NoInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        auto* ptr = new uint8_t[size];
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        delete[] ptr;  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("new[] (no init)");
}

// Benchmark raw new[] with value initialization (zero-init)
static void BM_New_ZeroInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        // Use () for value initialization which zero-initializes
        auto* ptr = new uint8_t[size]();
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        delete[] ptr;  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("new[] (zero-init)");
}

// Benchmark calloc (allocates and zero-initializes)
// calloc is a C function that allocates memory and initializes it to zero
static void BM_Calloc_ZeroInit(benchmark::State& state) {
    const auto size = static_cast<size_t>(state.range(0));

    for (auto _ : state) {
        void* ptr = std::calloc(size, 1);  // Allocate size bytes, zero-initialized
        benchmark::DoNotOptimize(ptr);
        benchmark::ClobberMemory();

        state.PauseTiming();
        std::free(ptr);  // Deallocation not timed
        state.ResumeTiming();
    }

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size));
    state.SetLabel("calloc (zero-init)");
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

BENCHMARK(BM_Vector_Reserve_NoInit)
    ->Apply(CustomArguments)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_Vector_Resize_ZeroInit)
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
