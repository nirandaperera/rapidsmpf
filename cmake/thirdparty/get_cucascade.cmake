# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Production builds only need topology discovery. Benchmark builds also use the cudf-free core
# library for the pinned-memory disk pipeline.
function(find_and_configure_cucascade)
  if(RAPIDSMPF_BUILD_BENCHMARKS)
    set(CPM_DOWNLOAD_cuCascade ON)
    set(cucascade_topology_only OFF)
    set(cucascade_global_targets cuCascade::cucascade_topology_discovery
                                 cuCascade::cucascade_static
    )
  else()
    set(cucascade_topology_only ON)
    set(cucascade_global_targets cuCascade::cucascade_topology_discovery)
  endif()

  rapids_cpm_find(
    cuCascade 0.1.0
    GLOBAL_TARGETS ${cucascade_global_targets}
    CPM_ARGS
    GIT_REPOSITORY https://github.com/NVIDIA/cuCascade.git
    GIT_TAG d515bb0536b8766bae61ec60a530df394467af64
    OPTIONS "CUCASCADE_BUILD_TESTS OFF"
            "CUCASCADE_BUILD_BENCHMARKS OFF"
            "CUCASCADE_BUILD_SHARED_LIBS OFF"
            "CUCASCADE_BUILD_STATIC_LIBS ON"
            "CUCASCADE_WARNINGS_AS_ERRORS OFF"
            "CUCASCADE_TOPOLOGY_ONLY ${cucascade_topology_only}"
            "CUCASCADE_BUILD_CUDF OFF"
            "CUCASCADE_BUILD_IO OFF"
    EXCLUDE_FROM_ALL ON
  )
endfunction()

find_and_configure_cucascade()
