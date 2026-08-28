/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>

#include <cucascade/data/disk_io_backend.hpp>
#include <cucascade/data/io_backend_registry.hpp>
#if RAPIDSMPF_HAVE_KVIKIO
#include <kvikio/compat_mode.hpp>
#include <kvikio/defaults.hpp>
#include <kvikio/file_handle.hpp>
#endif
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#include <rmm/cuda_stream_view.hpp>

namespace {

enum class backend_kind {
    cucascade,
    kvikio_auto,
    kvikio_gds
};
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
    backend_kind backend;
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

std::string_view name(backend_kind value) {
    switch (value) {
    case backend_kind::cucascade:
        return "cuCascade";
    case backend_kind::kvikio_auto:
        return "KvikIO_AUTO";
    case backend_kind::kvikio_gds:
        return "KvikIO_GDS";
    }
    std::terminate();
}

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

bool nvidia_fs_loaded() {
    std::ifstream modules{"/proc/modules"};
    std::string module;
    while (modules >> module) {
        if (module == "nvidia_fs") {
            return true;
        }
        std::string rest;
        std::getline(modules, rest);
    }
    return false;
}

void cuda_check(cudaError_t status, std::string_view action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string{action} + ": " + cudaGetErrorString(status));
    }
}

std::filesystem::path spill_directory() {
    if (auto const* value = std::getenv("RAPIDSMPF_DISK_SPILL_DIR");
        value != nullptr && *value != '\0')
    {
        return value;
    }
    return std::filesystem::temp_directory_path();
}

std::filesystem::path unique_path(std::string_view purpose) {
    auto const id = next_path_id.fetch_add(1, std::memory_order_relaxed);
    return spill_directory()
           / ("rapidsmpf-disk-spill-" + std::to_string(::getpid()) + "-"
              + std::to_string(id) + "-" + std::string{purpose} + ".bin");
}

void synchronize_file(std::filesystem::path const& path) {
    auto const fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            "open for fdatasync failed: " + std::string{std::strerror(errno)}
        );
    }
    if (::fdatasync(fd) != 0) {
        auto const error = std::string{std::strerror(errno)};
        ::close(fd);
        throw std::runtime_error("fdatasync failed: " + error);
    }
    if (::close(fd) != 0) {
        throw std::runtime_error(
            "close after fdatasync failed: " + std::string{std::strerror(errno)}
        );
    }
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

class adapter {
  public:
    explicit adapter(case_config const& config) : config_{config} {
        if (config.backend == backend_kind::cucascade) {
            cucascade::io_backend_registry registry;
            cucascade::register_builtin_io_backends(registry);
            cucascade_ = registry.create_backend("pipeline");
        }
    }

    [[nodiscard]] std::string mode_label() const {
#if RAPIDSMPF_HAVE_KVIKIO
        if (config_.backend == backend_kind::kvikio_auto
            || config_.backend == backend_kind::kvikio_gds)
        {
            if (config_.source != source_kind::device) {
                return "compat-host";
            }
            auto const mode = config_.backend == backend_kind::kvikio_auto
                                  ? kvikio::CompatMode::AUTO
                                  : kvikio::CompatMode::OFF;
            return kvikio::defaults::is_compat_mode_preferred(mode) || !nvidia_fs_loaded()
                       ? "compat"
                       : "gds";
        }
#endif
        return "n/a";
    }

    void write(
        std::filesystem::path const& path,
        data_buffer const& source,
        std::size_t file_offset
    ) {
        switch (config_.backend) {
        case backend_kind::cucascade:
            cucascade_write(path, source, file_offset);
            break;
        case backend_kind::kvikio_auto:
            kvikio_write(path, source, file_offset, false);
            break;
        case backend_kind::kvikio_gds:
            kvikio_write(path, source, file_offset, true);
            break;
        }
        if (config_.durable) {
            synchronize_file(path);
        }
    }

    void read(
        std::filesystem::path const& path,
        data_buffer& destination,
        std::size_t file_offset
    ) {
        switch (config_.backend) {
        case backend_kind::cucascade:
            cucascade_read(path, destination, file_offset);
            break;
        case backend_kind::kvikio_auto:
            kvikio_read(path, destination, file_offset, false);
            break;
        case backend_kind::kvikio_gds:
            kvikio_read(path, destination, file_offset, true);
            break;
        }
    }

  private:
    void cucascade_write(
        std::filesystem::path const& path,
        data_buffer const& source,
        std::size_t file_offset
    ) {
        if (source.kind() == source_kind::device) {
            cucascade_->write(
                path, source.data(), source.size(), file_offset, rmm::cuda_stream_default
            );
        } else {
            cucascade_->write(path, source.data(), source.size(), file_offset);
        }
    }

    void cucascade_read(
        std::filesystem::path const& path,
        data_buffer& destination,
        std::size_t file_offset
    ) {
        if (destination.kind() == source_kind::device) {
            cucascade_->read(
                path,
                destination.data(),
                destination.size(),
                file_offset,
                rmm::cuda_stream_default
            );
        } else {
            cucascade_->read(path, destination.data(), destination.size(), file_offset);
        }
    }

    void kvikio_write(
        std::filesystem::path const& path,
        data_buffer const& source,
        std::size_t file_offset,
        bool gds_only
    ) {
#if RAPIDSMPF_HAVE_KVIKIO
        auto const mode = gds_only ? kvikio::CompatMode::OFF : kvikio::CompatMode::AUTO;
        kvikio::FileHandle file(path.string(), "w+", kvikio::FileHandle::m644, mode);
        auto const written = file.pwrite(source.data(), source.size(), file_offset).get();
        if (written != source.size()) {
            throw std::runtime_error("KvikIO pwrite returned a short byte count");
        }
        file.close();
#else
        static_cast<void>(path);
        static_cast<void>(source);
        static_cast<void>(file_offset);
        static_cast<void>(gds_only);
        throw std::runtime_error("KvikIO was not found at configure time");
#endif
    }

    void kvikio_read(
        std::filesystem::path const& path,
        data_buffer& destination,
        std::size_t file_offset,
        bool gds_only
    ) {
#if RAPIDSMPF_HAVE_KVIKIO
        auto const mode = gds_only ? kvikio::CompatMode::OFF : kvikio::CompatMode::AUTO;
        kvikio::FileHandle file(path.string(), "r", kvikio::FileHandle::m644, mode);
        auto const bytes_read =
            file.pread(destination.data(), destination.size(), file_offset).get();
        if (bytes_read != destination.size()) {
            throw std::runtime_error("KvikIO pread returned a short byte count");
        }
        file.close();
#else
        static_cast<void>(path);
        static_cast<void>(destination);
        static_cast<void>(file_offset);
        static_cast<void>(gds_only);
        throw std::runtime_error("KvikIO was not found at configure time");
#endif
    }

    case_config config_;
    std::shared_ptr<cucascade::idisk_io_backend> cucascade_;
};

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

void perform_transfer(
    case_config const& config,
    adapter& backend,
    data_buffer const& source,
    data_buffer* destination,
    std::filesystem::path const& path
) {
    auto const file_offset = config.unaligned ? 1U : 0U;
    backend.write(path, source, file_offset);
    if (config.operation == operation_kind::round_trip) {
        backend.read(path, *destination, file_offset);
    }
}

void validate(case_config const& config) {
    auto const path = unique_path("validation");
    try {
        data_buffer source{config.source, config.size, config.unaligned, true};
        auto destination = config.operation == operation_kind::round_trip
                               ? std::make_unique<data_buffer>(
                                     config.source, config.size, config.unaligned, false
                                 )
                               : nullptr;
        adapter backend{config};
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

std::string unsupported_reason(case_config const& config) {
#if !RAPIDSMPF_HAVE_KVIKIO
    if (config.backend == backend_kind::kvikio_auto
        || config.backend == backend_kind::kvikio_gds)
    {
        return "KvikIO was not found at configure time";
    }
#endif
    if (config.backend == backend_kind::kvikio_gds) {
        if (config.source != source_kind::device) {
            return "GDS-only cases require a device-memory source";
        }
#if RAPIDSMPF_HAVE_KVIKIO
        if (!nvidia_fs_loaded()) {
            return "GDS unavailable; nvidia_fs kernel module is not loaded";
        }
        if (kvikio::defaults::is_compat_mode_preferred(kvikio::CompatMode::AUTO)) {
            return "GDS unavailable; KvikIO AUTO selected compatibility mode";
        }
#endif
    }
    if (config.source != source_kind::pageable_host
        || config.backend == backend_kind::cucascade)
    {
        std::string reason;
        if (!have_cuda_device(reason)) {
            return reason;
        }
    }
    return {};
}

void run_case(benchmark::State& state, case_config config) {
    auto const unsupported = unsupported_reason(config);
    if (!unsupported.empty()) {
        state.SkipWithMessage(unsupported);
        return;
    }

    try {
        validate(config);
    } catch (std::exception const& error) {
        state.SkipWithError(error.what());
        return;
    }

    std::vector<std::unique_ptr<data_buffer>> sources;
    std::vector<std::unique_ptr<data_buffer>> destinations;
    std::vector<std::unique_ptr<adapter>> adapters;
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
        adapters.push_back(std::make_unique<adapter>(config));
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
                        *adapters[static_cast<std::size_t>(i)],
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
    state.SetLabel(
        "mode=" + adapters.front()->mode_label()
        + (config.durable ? ",fdatasync" : ",page-cache")
    );
}

std::string benchmark_name(case_config const& config) {
    return std::string{name(config.backend)} + "/" + std::string{name(config.source)}
           + "/" + std::string{name(config.operation)} + "/" + std::to_string(config.size)
           + "B/" + (config.unaligned ? "unaligned" : "aligned") + "/c"
           + std::to_string(config.concurrency)
           + (config.durable ? "/durable" : "/ephemeral");
}

void register_case(case_config const& config) {
    benchmark::RegisterBenchmark(
        benchmark_name(config),
        [config](benchmark::State& state) { run_case(state, config); }
    )
        ->Iterations(1)
        ->UseRealTime();
}

void register_benchmarks() {
    if (smoke_test_mode()) {
        register_case(
            {backend_kind::cucascade,
             source_kind::device,
             operation_kind::round_trip,
             1 * mib,
             false,
             1,
             false}
        );
        register_case(
            {backend_kind::kvikio_auto,
             source_kind::device,
             operation_kind::round_trip,
             1 * mib,
             true,
             1,
             false}
        );
        register_case(
            {backend_kind::kvikio_auto,
             source_kind::pageable_host,
             operation_kind::round_trip,
             1 * mib,
             true,
             1,
             false}
        );
        register_case(
            {backend_kind::kvikio_gds,
             source_kind::device,
             operation_kind::write,
             4 * kib,
             false,
             1,
             false}
        );
        return;
    }

    constexpr backend_kind backends[]{
        backend_kind::cucascade, backend_kind::kvikio_auto, backend_kind::kvikio_gds
    };
    constexpr source_kind sources[]{
        source_kind::device, source_kind::pinned_host, source_kind::pageable_host
    };
    constexpr operation_kind operations[]{
        operation_kind::write, operation_kind::round_trip
    };
    constexpr std::size_t sizes[]{4 * kib, 1 * mib, 16 * mib, 64 * mib, 512 * mib};
    constexpr bool unaligned_values[]{false, true};
    constexpr int concurrency_values[]{1, 4};

    for (auto const backend : backends) {
        for (auto const source : sources) {
            for (auto const operation : operations) {
                for (auto const size : sizes) {
                    for (auto const unaligned : unaligned_values) {
                        for (auto const concurrency : concurrency_values) {
                            register_case(
                                {backend,
                                 source,
                                 operation,
                                 size,
                                 unaligned,
                                 concurrency,
                                 false}
                            );
                        }
                    }
                }
            }
        }
    }

    for (auto const backend : {backend_kind::cucascade, backend_kind::kvikio_auto}) {
        for (auto const source : sources) {
            register_case(
                {backend, source, operation_kind::write, 16 * mib, false, 1, true}
            );
        }
    }
}

auto const registered = []() {
    register_benchmarks();
    return true;
}();

}  // namespace

BENCHMARK_MAIN();
