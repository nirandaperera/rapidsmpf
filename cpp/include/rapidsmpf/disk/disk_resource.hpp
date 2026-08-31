/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <filesystem>

#include <rapidsmpf/config.hpp>

namespace rapidsmpf::disk {

/**
 * @brief Blocking, non-stream-ordered disk I/O for host or device byte buffers.
 *
 * Uses KvikIO with CompatMode::AUTO (GDS when available, POSIX/compat otherwise).
 *
 * Callers must synchronize any CUDA stream that produced or consumes a device
 * pointer before calling write() or read(). Each call completes before return,
 * so the pointer may be reused or freed afterward.
 *
 * Disk I/O is intentionally outside the MemoryType / BufferResource taxonomy.
 */
class DiskResource {
  public:
    DiskResource() = default;
    ~DiskResource() = default;

    DiskResource(DiskResource const&) = delete;
    DiskResource& operator=(DiskResource const&) = delete;
    DiskResource(DiskResource&&) = delete;
    DiskResource& operator=(DiskResource&&) = delete;

    /**
     * @brief Write bytes to a file.
     *
     * @param path File path.
     * @param data Host or device pointer to the source bytes.
     * @param size Number of bytes to write.
     * @param is_device True if @p data is a device pointer; false for host
     *                  (pinned or pageable).
     * @param file_offset Byte offset within the file.
     */
    void write(
        std::filesystem::path const& path,
        void const* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    );

    /**
     * @brief Read bytes from a file.
     *
     * @param path File path.
     * @param data Host or device pointer to the destination buffer.
     * @param size Number of bytes to read.
     * @param is_device True if @p data is a device pointer; false for host
     *                  (pinned or pageable).
     * @param file_offset Byte offset within the file.
     */
    void read(
        std::filesystem::path const& path,
        void* data,
        std::size_t size,
        bool is_device,
        std::size_t file_offset = 0
    );

    /**
     * @brief Durably synchronize file data to storage (fdatasync).
     *
     * Not used on the default spill path; exposed for benchmark durability cases.
     *
     * @param path File path.
     */
    void flush(std::filesystem::path const& path);
};

/**
 * @brief Spill directory from `disk_spill_dir` (`RAPIDSMPF_DISK_SPILL_DIR`).
 *
 * An empty option uses the system temporary directory.
 *
 * @param options Configuration options.
 * @return Directory used for spill files.
 */
[[nodiscard]] std::filesystem::path default_spill_directory(config::Options options);

}  // namespace rapidsmpf::disk
