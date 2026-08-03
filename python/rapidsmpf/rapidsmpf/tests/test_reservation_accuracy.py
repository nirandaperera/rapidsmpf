# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import rmm
import rmm.mr

from rapidsmpf.memory.buffer import MemoryType
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.memory.memory_reservation import opaque_memory_usage
from rapidsmpf.memory.reservation_accuracy import (
    allocation_scope,
    clear,
    get_samples,
    report,
    set_enabled,
)


KIB = 1024


def test_reservation_accuracy_same_thread_under_reserve() -> None:
    set_enabled(True)
    clear()
    try:
        br = BufferResource(rmm.mr.CudaMemoryResource())
        mr = br.device_mr_adaptor()
        reserved = 1 * KIB
        res, _ = br.reserve(MemoryType.DEVICE, reserved, allow_overbooking=False)

        with opaque_memory_usage(res, label="under"):
            buf = rmm.DeviceBuffer(size=4 * KIB, mr=mr)
            del buf

        samples = get_samples()
        assert len(samples) == 1
        sample = samples[0]
        assert sample.label == "under"
        assert sample.reserved == reserved
        assert sample.peak >= 4 * KIB
        assert sample.under_reserved
        assert sample.ratio >= 4.0
        assert "under-reserved" in report()
    finally:
        clear()
        set_enabled(None)


def test_reservation_accuracy_worker_thread_peak() -> None:
    set_enabled(True)
    clear()
    try:
        br = BufferResource(rmm.mr.CudaMemoryResource())
        mr = br.device_mr_adaptor()
        reserved = 8 * KIB
        res, _ = br.reserve(MemoryType.DEVICE, reserved, allow_overbooking=False)

        def worker() -> None:
            with allocation_scope():
                buf = rmm.DeviceBuffer(size=2 * KIB, mr=mr)
                del buf

        with opaque_memory_usage(res, label="worker"):
            ctx = contextvars.copy_context()
            with ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(ctx.run, worker).result()

        samples = get_samples()
        assert len(samples) == 1
        sample = samples[0]
        assert sample.label == "worker"
        assert sample.reserved == reserved
        assert sample.peak >= 2 * KIB
        assert not sample.under_reserved
        assert sample.over_bytes >= reserved - sample.peak
    finally:
        clear()
        set_enabled(None)


def test_reservation_accuracy_disabled_by_default() -> None:
    set_enabled(None)
    clear()
    br = BufferResource(rmm.mr.CudaMemoryResource())
    mr = br.device_mr_adaptor()
    res, _ = br.reserve(MemoryType.DEVICE, KIB, allow_overbooking=False)
    with opaque_memory_usage(res, label="disabled"):
        buf = rmm.DeviceBuffer(size=KIB, mr=mr)
        del buf
    assert get_samples() == []
