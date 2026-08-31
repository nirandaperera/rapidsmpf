/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <unistd.h>

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/io_backend_registry.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/disk/disk_resource.hpp>

namespace {

using rapidsmpf::disk::default_spill_directory;
using rapidsmpf::disk::DiskResource;

enum class source_kind {
    device,
    pinned_host,
    pageable_host
};
enum class operation_kind {
    write,
    round_trip
};

struct case_config {
    source_kind source;
    operation_kind operation;
    std::size_t size;
    bool unaligned;
    int concurrency;
    bool durable;
};

constexpr std::size_t kib = 1024;
constexpr std::size_t mib = 1024 * kib;
std::atomic<std::uint64_t> next_path_id{0};

class cucascade_disk {
  public:
    cucascade_disk() {
        cucascade::io_backend_registry registry;
        cucascade::register_builtin_io_backends(registry);
        backend_ = registry.create_backend("pipeline");
    }

    void write(
        std::filesystem::path const& path,
        void const* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    ) {
        if (is_device) {
            backend_->write(path, data, size, file_offset, rmm::cuda_stream_default);
        } else {
            backend_->write(path, data, size, file_offset);
        }
    }

    void read(
        std::filesystem::path const& path,
        void* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    ) {
        if (is_device) {
            backend_->read(path, data, size, file_offset, rmm::cuda_stream_default);
        } else {
            backend_->read(path, data, size, file_offset);
        }
    }

    void flush(std::filesystem::path const& path) {
        DiskResource{}.flush(path);
    }

    [[nodiscard]] static constexpr std::string_view backend_name() {
        return "cuCascade";
    }

  private:
    std::shared_ptr<cucascade::idisk_io_backend> backend_;
};

class kvikio_disk {
  public:
    void write(
        std::filesystem::path const& path,
        void const* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    ) {
        resource_.write(path, data, size, is_device, file_offset);
    }

    void read(
        std::filesystem::path const& path,
        void* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    ) {
        resource_.read(path, data, size, is_device, file_offset);
    }

    void flush(std::filesystem::path const& path) {
        resource_.flush(path);
    }

    [[nodiscard]] static constexpr std::string_view backend_name() {
        return "KvikIO_AUTO";
    }

  private:
    DiskResource resource_;
};

std::string_view name(source_kind value) {
    switch (value) {
    case source_kind::device:
        return "device";
    case source_kind::pinned_host:
        return "pinned";
    case source_kind::pageable_host:
        return "pageable";
    }
    std::terminate();
}

std::string_view name(operation_kind value) {
    return value == operation_kind::write ? "write" : "round_trip";
}

bool smoke_test_mode() {
    auto const* value = std::getenv("RAPIDSMPF_SMOKE_TEST_MODE");
    if (value == nullptr) {
        return false;
    }
    auto text = std::string{value};
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text == "1" || text == "on" || text == "true" || text == "yes";
}

bool have_cuda_device(std::string& reason) {
    int count = 0;
    auto const status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
        reason = status == cudaSuccess
                     ? "no CUDA device"
                     : std::string{"CUDA unavailable: "} + cudaGetErrorString(status);
        cudaGetLastError();
        return false;
    }
    return true;
}

void cuda_check(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + ": " + cudaGetErrorString(status));
    }
}

std::filesystem::path unique_path(std::string_view purpose) {
    auto const id = next_path_id.fetch_add(1, std::memory_order_relaxed);
    return default_spill_directory(
               rapidsmpf::config::Options{rapidsmpf::config::get_environment_variables()}
           )
           / ("rapidsmpf-disk-spill-" + std::to_string(::getpid()) + "-"
              + std::to_string(id) + "-" + std::string{purpose} + ".bin");
}

class data_buffer {
  public:
    data_buffer(source_kind kind, std::size_t size, bool unaligned, bool initialize)
        : kind_{kind}, size_{size}, offset_{unaligned ? 1U : 0U} {
        auto const allocation_size = size + offset_;
        if (kind == source_kind::pageable_host) {
            pageable_.resize(allocation_size);
            pointer_ = pageable_.data() + offset_;
        } else if (kind == source_kind::pinned_host) {
            cuda_check(
                cudaHostAlloc(&allocation_, allocation_size, cudaHostAllocPortable),
                "cudaHostAlloc"
            );
            pointer_ = static_cast<std::byte*>(allocation_) + offset_;
        } else {
            cuda_check(cudaMalloc(&allocation_, allocation_size), "cudaMalloc");
            pointer_ = static_cast<std::byte*>(allocation_) + offset_;
        }

        if (initialize) {
            auto expected = pattern();
            if (kind == source_kind::device) {
                cuda_check(
                    cudaMemcpy(pointer_, expected.data(), size_, cudaMemcpyHostToDevice),
                    "initial host-to-device copy"
                );
            } else {
                std::memcpy(pointer_, expected.data(), size_);
            }
        } else if (kind == source_kind::device) {
            cuda_check(cudaMemset(pointer_, 0, size_), "cudaMemset");
        } else {
            std::memset(pointer_, 0, size_);
        }
    }

    data_buffer(data_buffer const&) = delete;
    data_buffer& operator=(data_buffer const&) = delete;

    ~data_buffer() {
        if (allocation_ != nullptr) {
            if (kind_ == source_kind::device) {
                cudaFree(allocation_);
            } else {
                cudaFreeHost(allocation_);
            }
        }
    }

    [[nodiscard]] void* data() {
        return pointer_;
    }

    [[nodiscard]] void const* data() const {
        return pointer_;
    }

    [[nodiscard]] source_kind kind() const {
        return kind_;
    }

    [[nodiscard]] std::size_t size() const {
        return size_;
    }

    [[nodiscard]] bool is_device() const {
        return kind_ == source_kind::device;
    }

    [[nodiscard]] std::vector<std::byte> to_host() const {
        std::vector<std::byte> result(size_);
        if (kind_ == source_kind::device) {
            cuda_check(
                cudaMemcpy(result.data(), pointer_, size_, cudaMemcpyDeviceToHost),
                "validation device-to-host copy"
            );
        } else {
            std::memcpy(result.data(), pointer_, size_);
        }
        return result;
    }

    [[nodiscard]] std::vector<std::byte> pattern() const {
        std::vector<std::byte> result(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            auto const value =
                static_cast<unsigned char>(((i * 131U) ^ (i >> 7U) ^ 0x5aU) & 0xffU);
            result[i] = static_cast<std::byte>(value);
        }
        return result;
    }

  private:
    source_kind kind_;
    std::size_t size_;
    std::size_t offset_;
    std::vector<std::byte> pageable_;
    void* allocation_{nullptr};
    void* pointer_{nullptr};
};

void sync_if_device(data_buffer const& buffer) {
    if (buffer.is_device()) {
        cuda_check(cudaStreamSynchronize(nullptr), "cudaStreamSynchronize");
    }
}

template <typename Backend>
void perform_transfer(
    case_config const& config,
    Backend& backend,
    data_buffer const& source,
    data_buffer* destination,
    std::filesystem::path const& path
) {
    auto const file_offset = config.unaligned ? 1U : 0U;
    sync_if_device(source);
    backend.write(path, source.data(), source.size(), source.is_device(), file_offset);
    if (config.durable) {
        backend.flush(path);
    }
    if (config.operation == operation_kind::round_trip) {
        sync_if_device(*destination);
        backend.read(
            path,
            destination->data(),
            destination->size(),
            destination->is_device(),
            file_offset
        );
    }
}

void remove_and_check(std::filesystem::path const& path) {
    std::error_code error;
    auto const removed = std::filesystem::remove(path, error);
    if (!removed || error || std::filesystem::exists(path)) {
        throw std::runtime_error("failed to clean benchmark file: " + path.string());
    }
}

void check_exact_size(std::filesystem::path const& path, std::size_t expected_size) {
    auto const actual = std::filesystem::file_size(path);
    if (actual != expected_size) {
        throw std::runtime_error(
            "file size mismatch: expected " + std::to_string(expected_size) + ", got "
            + std::to_string(actual)
        );
    }
}

void check_file_contents(
    std::filesystem::path const& path,
    std::size_t file_offset,
    std::vector<std::byte> const& expected
) {
    std::vector<std::byte> actual(expected.size());
    std::ifstream file{path, std::ios::binary};
    file.seekg(static_cast<std::streamoff>(file_offset));
    file.read(
        reinterpret_cast<char*>(actual.data()),
        static_cast<std::streamsize>(actual.size())
    );
    if (!file || file.gcount() != static_cast<std::streamsize>(actual.size())
        || actual != expected)
    {
        throw std::runtime_error("file content validation failed");
    }
}

template <typename Backend>
void validate(case_config const& config) {
    auto const path = unique_path("validation");
    try {
        data_buffer source{config.source, config.size, config.unaligned, true};
        auto destination = config.operation == operation_kind::round_trip
                               ? std::make_unique<data_buffer>(
                                     config.source, config.size, config.unaligned, false
                                 )
                               : nullptr;
        Backend backend{};
        perform_transfer(config, backend, source, destination.get(), path);
        auto const file_offset = config.unaligned ? 1U : 0U;
        auto const expected = source.pattern();
        check_exact_size(path, config.size + file_offset);
        check_file_contents(path, file_offset, expected);
        if (destination != nullptr && destination->to_host() != expected) {
            throw std::runtime_error("round-trip content validation failed");
        }
        remove_and_check(path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
        throw;
    }
}

std::string case_unsupported_reason(case_config const& config) {
    if (config.source != source_kind::pageable_host) {
        std::string cuda_reason;
        if (!have_cuda_device(cuda_reason)) {
            return cuda_reason;
        }
    }
    return {};
}

template <typename Backend>
void run_case(benchmark::State& state, case_config config) {
    auto const unsupported = case_unsupported_reason(config);
    if (!unsupported.empty()) {
        state.SkipWithMessage(unsupported);
        return;
    }

    try {
        validate<Backend>(config);
    } catch (std::exception const& error) {
        state.SkipWithError(error.what());
        return;
    }

    std::vector<std::unique_ptr<data_buffer>> sources;
    std::vector<std::unique_ptr<data_buffer>> destinations;
    std::vector<std::unique_ptr<Backend>> backends;
    for (int i = 0; i < config.concurrency; ++i) {
        sources.push_back(
            std::make_unique<data_buffer>(
                config.source, config.size, config.unaligned, true
            )
        );
        if (config.operation == operation_kind::round_trip) {
            destinations.push_back(
                std::make_unique<data_buffer>(
                    config.source, config.size, config.unaligned, false
                )
            );
        }
        backends.push_back(std::make_unique<Backend>());
    }

    for (auto _ : state) {
        static_cast<void>(_);
        state.PauseTiming();
        std::vector<std::filesystem::path> paths;
        for (int i = 0; i < config.concurrency; ++i) {
            paths.push_back(unique_path("timed"));
        }
        std::exception_ptr thread_error;
        std::mutex error_mutex;
        std::barrier start{config.concurrency};
        std::vector<std::thread> workers;
        state.ResumeTiming();

        for (int i = 0; i < config.concurrency; ++i) {
            workers.emplace_back([&, i]() {
                start.arrive_and_wait();
                try {
                    perform_transfer(
                        config,
                        *backends[static_cast<std::size_t>(i)],
                        *sources[static_cast<std::size_t>(i)],
                        config.operation == operation_kind::round_trip
                            ? destinations[static_cast<std::size_t>(i)].get()
                            : nullptr,
                        paths[static_cast<std::size_t>(i)]
                    );
                } catch (...) {
                    std::lock_guard lock{error_mutex};
                    if (thread_error == nullptr) {
                        thread_error = std::current_exception();
                    }
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }

        state.PauseTiming();
        for (auto const& path : paths) {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
        if (thread_error != nullptr) {
            std::rethrow_exception(thread_error);
        }
        state.ResumeTiming();
    }

    auto const operation_factor =
        config.operation == operation_kind::round_trip ? 2U : 1U;
    state.SetBytesProcessed(
        static_cast<std::int64_t>(state.iterations())
        * static_cast<std::int64_t>(config.concurrency)
        * static_cast<std::int64_t>(operation_factor)
        * static_cast<std::int64_t>(config.size)
    );
    state.SetLabel(config.durable ? "fdatasync" : "page-cache");
}

template <typename Backend>
std::string benchmark_name(case_config const& config) {
    return std::string{Backend::backend_name()} + "/" + std::string{name(config.source)}
           + "/" + std::string{name(config.operation)} + "/" + std::to_string(config.size)
           + "B/" + (config.unaligned ? "unaligned" : "aligned") + "/c"
           + std::to_string(config.concurrency)
           + (config.durable ? "/durable" : "/ephemeral");
}

template <typename Backend>
void register_case(case_config const& config) {
    benchmark::RegisterBenchmark(
        benchmark_name<Backend>(config),
        [config](benchmark::State& state) { run_case<Backend>(state, config); }
    )
        ->Iterations(1)
        ->UseRealTime();
}

template <typename Backend>
void register_backend_cases() {
    constexpr source_kind sources[]{
        source_kind::device, source_kind::pinned_host, source_kind::pageable_host
    };
    constexpr operation_kind operations[]{
        operation_kind::write, operation_kind::round_trip
    };
    constexpr std::size_t sizes[]{4 * kib, 1 * mib, 16 * mib, 64 * mib, 512 * mib};
    constexpr bool unaligned_values[]{false, true};
    constexpr int concurrency_values[]{1, 4};

    for (auto const source : sources) {
        for (auto const operation : operations) {
            for (auto const size : sizes) {
                for (auto const unaligned : unaligned_values) {
                    for (auto const concurrency : concurrency_values) {
                        register_case<Backend>(
                            {source, operation, size, unaligned, concurrency, false}
                        );
                    }
                }
            }
        }
    }

    for (auto const source : sources) {
        register_case<Backend>({source, operation_kind::write, 16 * mib, false, 1, true});
    }
}

void register_benchmarks() {
    if (smoke_test_mode()) {
        register_case<cucascade_disk>(
            {source_kind::device, operation_kind::round_trip, 1 * mib, false, 1, false}
        );
        register_case<kvikio_disk>(
            {source_kind::device, operation_kind::round_trip, 1 * mib, true, 1, false}
        );
        register_case<kvikio_disk>(
            {source_kind::pageable_host,
             operation_kind::round_trip,
             1 * mib,
             true,
             1,
             false}
        );
        return;
    }

    register_backend_cases<cucascade_disk>();
    register_backend_cases<kvikio_disk>();
}

auto const registered = []() {
    register_benchmarks();
    return true;
}();

}  // namespace

BENCHMARK_MAIN();
