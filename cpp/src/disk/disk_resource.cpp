/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rapidsmpf/disk/disk_resource.hpp>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <unistd.h>

#include <kvikio/compat_mode.hpp>
#include <kvikio/file_handle.hpp>

#include <rapidsmpf/config.hpp>
#include <rapidsmpf/error.hpp>

namespace rapidsmpf::disk {

namespace {

void synchronize_file(std::filesystem::path const& path) {
    auto const fd = ::open(path.c_str(), O_RDONLY);
    RAPIDSMPF_EXPECTS(
        fd >= 0,
        "open for fdatasync failed: " + std::string{std::strerror(errno)},
        std::runtime_error
    );
    if (::fdatasync(fd) != 0) {
        auto const error = std::string{std::strerror(errno)};
        ::close(fd);
        RAPIDSMPF_FAIL("fdatasync failed: " + error, std::runtime_error);
    }
    RAPIDSMPF_EXPECTS(
        ::close(fd) == 0,
        "close after fdatasync failed: " + std::string{std::strerror(errno)},
        std::runtime_error
    );
}

}  // namespace

void DiskResource::write(
    std::filesystem::path const& path,
    void const* data,
    std::size_t size,
    bool is_device,
    std::size_t file_offset
) {
    static_cast<void>(is_device);
    kvikio::FileHandle file(
        path.string(), "w+", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    );
    auto const written = file.pwrite(data, size, file_offset).get();
    RAPIDSMPF_EXPECTS(
        written == size, "KvikIO pwrite returned a short byte count", std::runtime_error
    );
    file.close();
}

void DiskResource::read(
    std::filesystem::path const& path,
    void* data,
    std::size_t size,
    bool is_device,
    std::size_t file_offset
) {
    static_cast<void>(is_device);
    kvikio::FileHandle file(
        path.string(), "r", kvikio::FileHandle::m644, kvikio::CompatMode::AUTO
    );
    auto const bytes_read = file.pread(data, size, file_offset).get();
    RAPIDSMPF_EXPECTS(
        bytes_read == size, "KvikIO pread returned a short byte count", std::runtime_error
    );
    file.close();
}

void DiskResource::flush(std::filesystem::path const& path) {
    synchronize_file(path);
}

std::filesystem::path default_spill_directory(config::Options options) {
    return options.get<std::filesystem::path>(
        "disk_spill_dir", [](std::string const& value) {
            if (value.empty()) {
                return std::filesystem::temp_directory_path();
            }
            return std::filesystem::path{value};
        }
    );
}

}  // namespace rapidsmpf::disk
