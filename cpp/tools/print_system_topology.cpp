/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @brief Print the system topology in a human-readable form.
 *
 * Runs cucascade topology discovery and prints GPUs, NUMA nodes, network devices,
 * storage devices, kTLS state, and the host-memory values RapidsMPF derives from them.
 *
 * Each process prints one self-contained report tagged with its rank (from rrun or
 * Slurm), so it can be launched on multiple ranks to compare per-rank bindings:
 *
 * Usage:
 *   print_system_topology
 *   rrun -n 4 print_system_topology
 */

#include <unistd.h>


#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <cucascade/memory/topology_discovery.hpp>

#include <rapidsmpf/bootstrap/utils.hpp>
#include <rapidsmpf/system_info.hpp>

namespace {

template <typename T>
std::string join(std::vector<T> const& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            s += ", ";
        }
        if constexpr (std::is_same_v<T, std::string>) {
            s += v[i];
        } else {
            s += std::to_string(v[i]);
        }
    }
    return s.empty() ? "-" : s;
}

double gib(std::uint64_t bytes) {
    return static_cast<double>(bytes) / static_cast<double>(1ULL << 30);
}

char const* drive_type(cucascade::memory::StorageDriveType t) {
    using cucascade::memory::StorageDriveType;
    switch (t) {
    case StorageDriveType::NVME:
        return "NVMe";
    case StorageDriveType::SATA_SSD:
        return "SATA SSD";
    case StorageDriveType::SATA_HDD:
        return "SATA HDD";
    default:
        return "unknown";
    }
}

}  // namespace

int main() {
    std::ostringstream out;
    cucascade::memory::topology_discovery discovery;
    if (!discovery.discover()) {
        std::cerr << "Failed to discover system topology" << std::endl;
        return 1;
    }
    auto const& topo = discovery.get_topology();

    int rank = -1;
    int nranks = -1;
    try {
        rank = rapidsmpf::bootstrap::get_rank();
        nranks = rapidsmpf::bootstrap::get_nranks();
    } catch (std::exception const&) {  // not launched by rrun/Slurm: single process
    }
    char host[256] = "?";
    ::gethostname(host, sizeof(host));
    int const gpu_id = rapidsmpf::bootstrap::get_gpu_id();
    std::string const ucx = rapidsmpf::bootstrap::get_ucx_net_devices();
    if (rank >= 0) {
        out << "=== [rank " << rank << "/" << nranks << "] ===\n";
    }
    out << "Process binding:\n"
        << "  pid:                  " << ::getpid() << " on " << host << "\n"
        << "  GPU id:               "
        << (gpu_id >= 0 ? std::to_string(gpu_id) : "(unset)") << "\n"
        << "  UCX_NET_DEVICES:      " << (ucx.empty() ? "(unset)" : ucx) << "\n"
        << "  CPU affinity:         "
        << rapidsmpf::bootstrap::get_current_cpu_affinity() << "\n"
        << "  NUMA memory nodes:    " << join(rapidsmpf::get_current_numa_nodes())
        << "\n\n";

    out << "Host: " << topo.hostname << "\n"
              << "GPUs: " << topo.num_gpus << ", NUMA nodes: " << topo.num_numa_nodes
              << ", network devices: " << topo.num_network_devices << "\n"
              << "kTLS: " << (topo.kernel_tls.loaded ? "available" : "not loaded")
              << (topo.kernel_tls.built_in ? " (built-in)" : "") << "\n";

    out << "\nGPUs:\n";
    for (auto const& g : topo.gpus) {
        out << "  [" << g.id << "] " << g.name << "\n"
                  << "      pci_bus_id:      " << g.pci_bus_id << "\n"
                  << "      uuid:            " << g.uuid << "\n"
                  << "      numa_node:       " << g.numa_node << "\n"
                  << "      cpu_affinity:    " << g.cpu_affinity_list << "\n"
                  << "      memory_binding:  " << join(g.memory_binding) << "\n"
                  << "      network_devices: " << join(g.network_devices) << "\n";
    }

    out << "\nNUMA nodes:\n";
    for (auto const& n : topo.numa_nodes) {
        out << "  [" << n.id << "] total=" << gib(n.memory_capacity)
                  << " GiB, free=" << gib(n.free_memory)
                  << " GiB, cpus=" << (n.has_cpus ? "yes" : "no")
                  << (n.is_device_memory ? ", device-memory" : "") << "\n";
    }

    out << "\nNetwork devices:\n";
    for (auto const& d : topo.network_devices) {
        out << "  " << d.name << ": numa=" << d.numa_node
                  << ", pci=" << d.pci_bus_id << ", iface="
                  << (d.net_interface.empty() ? "-" : d.net_interface)
                  << ", mtu=" << d.mtu << "\n";
    }

    out << "\nStorage devices:\n";
    for (auto const& d : topo.storage_devices) {
        out << "  " << d.name << ": " << drive_type(d.type)
                  << ", numa=" << d.numa_node << ", pci=" << d.pci_bus_id << "\n";
    }

    out << "\nRapidsMPF derived values:\n"
              << "  total host memory:      " << gib(rapidsmpf::get_total_host_memory())
              << " GiB\n"
              << "  current NUMA node:      " << rapidsmpf::get_current_numa_node() << "\n";
    try {
        out << "  host memory per GPU:    "
                  << gib(rapidsmpf::get_host_memory_per_gpu()) << " GiB\n";
    } catch (std::exception const& e) {
        out << "  host memory per GPU:    unavailable (" << e.what() << ")\n";
    }
    // Single write so reports from different ranks don't interleave.
    std::string const report = out.str();
    std::cout << report << std::flush;
    return 0;
}
