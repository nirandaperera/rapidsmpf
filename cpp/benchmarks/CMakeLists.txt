# =================================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =================================================================================

add_executable(bench_comm "bench_comm.cpp")
set_target_properties(
  bench_comm
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             # For std:: support of __int128_t. Can be removed once using cuda::std
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_comm PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                     "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_comm PRIVATE rapidsmpf::rapidsmpf ucxx::ucxx $<TARGET_NAME_IF_EXISTS:MPI::MPI_CXX>
                     $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
install(
  TARGETS bench_comm
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

# Find or install GoogleBench
include(${rapids-cmake-dir}/cpm/gbench.cmake)
rapids_cpm_gbench(BUILD_STATIC)

find_package(kvikio QUIET CONFIG)

add_executable(bench_disk_spill "bench_disk_spill.cpp")
set_target_properties(
  bench_disk_spill
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
)
target_compile_options(bench_disk_spill PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>")
target_compile_definitions(bench_disk_spill PRIVATE RAPIDSMPF_HAVE_KVIKIO=$<BOOL:${kvikio_FOUND}>)
target_link_libraries(
  bench_disk_spill
  PRIVATE cuCascade::cucascade_static rmm::rmm benchmark::benchmark
          $<TARGET_NAME_IF_EXISTS:kvikio::kvikio> $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
install(
  TARGETS bench_disk_spill
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)

add_executable(bench_memory_resources "bench_memory_resources.cpp")
set_target_properties(
  bench_memory_resources
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${RAPIDSMPF_BINARY_DIR}/benchmarks"
             CXX_STANDARD 20
             CXX_STANDARD_REQUIRED ON
             CXX_EXTENSIONS ON
             CUDA_STANDARD 20
             CUDA_STANDARD_REQUIRED ON
)
target_compile_options(
  bench_memory_resources PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAPIDSMPF_CXX_FLAGS}>"
                                 "$<$<COMPILE_LANGUAGE:CUDA>:${RAPIDSMPF_CUDA_FLAGS}>"
)
target_link_libraries(
  bench_memory_resources
  PRIVATE rapidsmpf::rapidsmpf rmm::rmm benchmark::benchmark benchmark::benchmark_main
          $<TARGET_NAME_IF_EXISTS:conda_env> maybe_asan
)
install(
  TARGETS bench_memory_resources
  COMPONENT benchmarking
  DESTINATION bin/benchmarks/librapidsmpf
  EXCLUDE_FROM_ALL
)
