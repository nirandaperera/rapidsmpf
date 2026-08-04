/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <algorithm>
#include <barrier>
#include <exception>
#include <future>
#include <limits>
#include <vector>

#include <cuda_runtime_api.h>

#include <cuda/memory_resource>

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/pinned_memory_resource.hpp>
#include <rapidsmpf/utils/misc.hpp>
#include <rapidsmpf/utils/string.hpp>

namespace rapidsmpf {

namespace {

cuda::memory_pool_properties get_memory_pool_properties(
    PinnedPoolProperties const& pool_properties
) {
    return cuda::memory_pool_properties{
        // if num_init_threads > 1 then we prime the pool in parallel, otherwise we use
        // CCCL's own priming
        .initial_pool_size =
            pool_properties.num_init_threads == 1 ? pool_properties.initial_pool_size : 0,
        // Before <https://github.com/NVIDIA/cccl/pull/6718>, the default
        // `release_threshold` was 0, which defeats the purpose of having a pool. We
        // now set it so the pool never releases unused pinned memory.
        .release_threshold = std::numeric_limits<std::size_t>::max(),
        // This defines how the allocations can be exported (IPC). See the docs of
        // `cudaMemPoolCreate` in <https://docs.nvidia.com/cuda/cuda-runtime-api>.
        .allocation_handle_type = ::cudaMemAllocationHandleType::cudaMemHandleTypeNone,
        .max_pool_size = pool_properties.max_pool_size.value_or(0),
    };
}

/**
 * @brief Prime a pinned memory pool in parallel.
 *
 * Faults in and registers the host pages backing roughly @p initial_pool_size
 * bytes, using @p num_init_threads threads that each allocate an equal chunk on
 * its own non-blocking CUDA stream. Only used when @p num_init_threads > 1 else, defaults
 * to  CCCL's own priming.
 *
 * @note No stream synchronization is performed, which is consistent with CCCL's priming.
 *
 * The chunk size is rounded up to `rmm::CUDA_ALLOCATION_ALIGNMENT`, so the total
 * may exceed @p initial_pool_size by up to `num_init_threads * alignment`.
 * Priming is skipped entirely when @p initial_pool_size is below the alignment.
 */
void prime_pinned_pool_parallel(
    detail::RmmResourceAdaptorImpl<cuda::pinned_memory_pool>& mr,
    std::size_t initial_pool_size,
    std::size_t num_init_threads
) {
    RAPIDSMPF_EXPECTS(
        num_init_threads >= 1, "num_init_threads must be >= 1", std::invalid_argument
    );

    constexpr std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT;
    if (initial_pool_size < alignment) {
        return;
    }

    std::size_t const per_chunk =
        rmm::align_up(initial_pool_size / num_init_threads, alignment);

    std::barrier barrier{safe_cast<std::ptrdiff_t>(num_init_threads)};
    std::vector<std::future<void>> futures;
    futures.reserve(num_init_threads);
    for (std::size_t i = 0; i < num_init_threads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&mr, &barrier, per_chunk]() {
            rmm::cuda_stream stream{rmm::cuda_stream::flags::non_blocking};
            void* ptr = nullptr;
            std::exception_ptr error;
            try {
                ptr = mr.allocate(stream.view(), per_chunk, alignment);
            } catch (...) {  // defer exception until all threads have allocated
                error = std::current_exception();
            }

            // wait for all threads to allocate
            barrier.arrive_and_wait();
            if (ptr != nullptr) {
                mr.deallocate(stream.view(), ptr, per_chunk, alignment);
            }
            // wait for all threads to deallocate
            barrier.arrive_and_wait();

            if (error) {  // rethrow exception if any thread failed to allocate
                std::rethrow_exception(error);
            }
        }));
    }

    for (auto& f : futures) {
        f.get();
    }
}

}  // namespace

PinnedMemoryResource::PinnedMemoryResource(PinnedPoolProperties pool_properties)
    : shared_base([&] {
          RAPIDSMPF_EXPECTS(
              is_pinned_memory_resources_supported(),
              "Pinned host memory is not supported on this system. "
              "CUDA " RAPIDSMPF_PINNED_MEM_RES_MIN_CUDA_VERSION_STR
              " is one of the requirements, but additional platform or driver "
              "constraints may apply. If needed, disable pinned host memory by passing "
              "`PinnedMemoryDisabled/ std::nullopt` for the `BufferResource` "
              "`pinned_pool_properties`, noting that this may significantly degrade "
              "spilling performance.",
              std::invalid_argument
          );
          RAPIDSMPF_EXPECTS(
              !pool_properties.max_pool_size.has_value()
                  || pool_properties.initial_pool_size <= *pool_properties.max_pool_size,
              "initial_pool_size must not exceed max_pool_size",
              std::invalid_argument
          );
          return cuda::mr::make_shared_resource<
              detail::RmmResourceAdaptorImpl<cuda::pinned_memory_pool>>(
              std::in_place,
              pool_properties.numa_id,
              get_memory_pool_properties(pool_properties)
          );
      }()),
      pool_properties_{std::move(pool_properties)} {
    if (pool_properties_.num_init_threads > 1) {
        prime_pinned_pool_parallel(
            get(), pool_properties_.initial_pool_size, pool_properties_.num_init_threads
        );
    }
}

std::optional<PinnedPoolProperties> pinned_pool_properties_from_options(
    config::Options options
) {
    bool const pinned_memory = options.get<bool>("pinned_memory", parse_string<bool>);
    if (!pinned_memory) {
        return PinnedMemoryDisabled;
    }

    auto const host_memory_per_gpu = get_host_memory_per_gpu();
    auto const total = safe_cast<double>(host_memory_per_gpu);
    return PinnedPoolProperties{
        .initial_pool_size = options.get<size_t>(
            "pinned_initial_pool_size",
            [total](auto const& s) { return parse_nbytes_or_percent(s, total); }
        ),
        .max_pool_size = options.get<std::optional<size_t>>(
            "pinned_max_pool_size",
            [total](auto const& s) { return parse_nbytes_or_percent(s, total); }
        ),
        .num_init_threads = options.get<std::size_t>(
            "pinned_memory_num_init_thread", [](std::string const& s) -> std::size_t {
                auto const v = parse_string<int>(s);
                RAPIDSMPF_EXPECTS(
                    v >= 1,
                    "pinned_memory_num_init_thread must be a positive integer",
                    std::invalid_argument
                );
                return safe_cast<std::size_t>(v);
            }
        ),
    };
}

}  // namespace rapidsmpf
