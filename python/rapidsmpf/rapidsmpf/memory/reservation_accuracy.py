# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Track soft-reservation accuracy against actual device allocations.

Enable with ``RAPIDSMPF_RESERVATION_ACCURACY=1`` (or :func:`set_enabled`).
When enabled, :func:`~rapidsmpf.memory.memory_reservation.opaque_memory_usage`
publishes an active tracking context and opens a thread-local scoped memory
record on the calling thread.

Offloaded work must run under a ``contextvars.copy_context()``-aware executor
and wrap the worker body with :func:`allocation_scope` so peak usage is
attributed to the thread that actually allocates. Without that wrap, only
same-thread allocations inside the opaque window are measured.

Each opaque window produces one sample comparing reserved size against the
max allocation peak observed while the reservation was held. A ratio
``peak / reserved > 1`` means the soft reservation under-booked and would OOM
under a hard reservation MR.
"""

from __future__ import annotations

import atexit
import contextvars
import os
import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rapidsmpf.memory.memory_reservation import MemoryReservation
    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

_ENV_FLAG = "RAPIDSMPF_RESERVATION_ACCURACY"


@dataclass(frozen=True, slots=True)
class AccuracySample:
    """One opaque-reservation window compared against observed allocations."""

    label: str
    reserved: int
    peak: int
    total_allocated: int
    net: int
    num_allocs: int

    @property
    def ratio(self) -> float:
        """Peak allocated bytes divided by reserved bytes."""
        if self.reserved == 0:
            return float("inf") if self.peak > 0 else 0.0
        return self.peak / self.reserved

    @property
    def under_bytes(self) -> int:
        """Bytes by which peak exceeded the reservation (0 if over-reserved)."""
        return max(0, self.peak - self.reserved)

    @property
    def over_bytes(self) -> int:
        """Unused reserved bytes relative to peak (0 if under-reserved)."""
        return max(0, self.reserved - self.peak)

    @property
    def under_reserved(self) -> bool:
        """True when peak usage exceeded the reservation."""
        return self.peak > self.reserved


@dataclass(slots=True)
class _ActiveTracking:
    label: str
    reserved: int
    mr: RmmResourceAdaptor
    worker_recorded: bool = field(default=False, compare=False)
    worker_peak: int = field(default=0, compare=False)
    worker_total: int = field(default=0, compare=False)
    worker_net: int = field(default=0, compare=False)
    worker_allocs: int = field(default=0, compare=False)


_active: contextvars.ContextVar[_ActiveTracking | None] = contextvars.ContextVar(
    "rapidsmpf_reservation_accuracy", default=None
)

_lock = threading.Lock()
_samples: list[AccuracySample] = []
_enabled_override: bool | None = None


def is_enabled() -> bool:
    """Return whether reservation-accuracy tracking is enabled."""
    if _enabled_override is not None:
        return _enabled_override
    return os.environ.get(_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def set_enabled(enabled: bool | None) -> None:
    """Override the env flag.

    Parameters
    ----------
    enabled
        ``True``/``False`` force on/off. ``None`` restores env-var control.
    """
    global _enabled_override
    _enabled_override = enabled


def clear() -> None:
    """Drop all collected samples."""
    with _lock:
        _samples.clear()


def get_samples() -> list[AccuracySample]:
    """Return a copy of collected samples."""
    with _lock:
        return list(_samples)


def _record(sample: AccuracySample) -> None:
    with _lock:
        _samples.append(sample)


def active_tracking() -> _ActiveTracking | None:
    """Return the active tracking context for this contextvars context, if any."""
    return _active.get()


@contextmanager
def track_opaque_reservation(
    reservation: MemoryReservation,
    *,
    label: str | None = None,
) -> Iterator[MemoryReservation]:
    """Publish reservation metadata for an opaque usage window.

    Opens a same-thread scoped memory record as a fallback for sync bodies.
    Prefer :func:`allocation_scope` on worker threads that perform the real
    allocations. One sample is emitted when the window exits.
    """
    if not is_enabled():
        yield reservation
        return

    tracking = _ActiveTracking(
        label=label or "opaque_memory_usage",
        reserved=reservation.size,
        mr=reservation.br.device_mr_adaptor(),
    )
    token = _active.set(tracking)
    tracking.mr.begin_scoped_memory_record()
    try:
        yield reservation
    finally:
        try:
            scope = tracking.mr.end_scoped_memory_record()
        finally:
            _active.reset(token)
        if tracking.worker_recorded:
            peak = max(tracking.worker_peak, int(scope.peak()))
            _record(
                AccuracySample(
                    label=tracking.label,
                    reserved=tracking.reserved,
                    peak=peak,
                    total_allocated=tracking.worker_total + int(scope.total()),
                    net=max(tracking.worker_net, int(scope.current())),
                    num_allocs=tracking.worker_allocs + int(scope.num_total_allocs()),
                )
            )
        else:
            _record(
                AccuracySample(
                    label=tracking.label,
                    reserved=tracking.reserved,
                    peak=int(scope.peak()),
                    total_allocated=int(scope.total()),
                    net=int(scope.current()),
                    num_allocs=int(scope.num_total_allocs()),
                )
            )


@contextmanager
def allocation_scope() -> Iterator[None]:
    """Accumulate peak device allocations for the active opaque reservation.

    No-op when accuracy tracking is disabled or no opaque reservation is
    active in the current contextvars context. Peaks from multiple worker
    scopes in the same opaque window are combined; the window emits one
    sample on exit.
    """
    tracking = _active.get()
    if tracking is None:
        yield
        return

    tracking.mr.begin_scoped_memory_record()
    try:
        yield
    finally:
        scope = tracking.mr.end_scoped_memory_record()
        tracking.worker_recorded = True
        peak = int(scope.peak())
        tracking.worker_peak = max(tracking.worker_peak, peak)
        tracking.worker_total += int(scope.total())
        tracking.worker_net = int(scope.current())
        tracking.worker_allocs += int(scope.num_total_allocs())


def summary(samples: list[AccuracySample] | None = None) -> list[dict[str, Any]]:
    """Aggregate samples by label."""
    data = samples if samples is not None else get_samples()
    by_label: dict[str, list[AccuracySample]] = defaultdict(list)
    for sample in data:
        by_label[sample.label].append(sample)

    rows: list[dict[str, Any]] = []
    for label, group in sorted(by_label.items()):
        peaks = [s.peak for s in group]
        reserved = [s.reserved for s in group]
        ratios = [s.ratio for s in group if s.reserved > 0]
        under = [s for s in group if s.under_reserved]
        rows.append(
            {
                "label": label,
                "calls": len(group),
                "under_reserved_calls": len(under),
                "under_reserved_pct": 100.0 * len(under) / len(group),
                "mean_reserved": sum(reserved) / len(group),
                "mean_peak": sum(peaks) / len(group),
                "max_peak": max(peaks),
                "max_under_bytes": max((s.under_bytes for s in group), default=0),
                "mean_ratio": (sum(ratios) / len(ratios)) if ratios else 0.0,
                "max_ratio": max(ratios) if ratios else 0.0,
            }
        )
    return rows


def report(samples: list[AccuracySample] | None = None) -> str:
    """Return a human-readable reservation-accuracy report."""
    data = samples if samples is not None else get_samples()
    if not data:
        return "Reservation accuracy: no samples collected."

    lines = [
        f"Reservation accuracy: {len(data)} samples "
        f"({sum(1 for s in data if s.under_reserved)} under-reserved)",
        "",
        f"{'label':<40} {'calls':>6} {'under%':>7} {'mean_ratio':>10} "
        f"{'max_ratio':>9} {'max_under':>12}",
        "-" * 90,
    ]
    for row in summary(data):
        lines.append(
            f"{row['label']:<40} {row['calls']:>6} {row['under_reserved_pct']:>6.1f}% "
            f"{row['mean_ratio']:>10.2f} {row['max_ratio']:>9.2f} "
            f"{_fmt_bytes(row['max_under_bytes']):>12}"
        )
    return "\n".join(lines)


def _fmt_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{n}B"


def _atexit_report() -> None:
    """Print any leftover samples (e.g. if per-query clear was not used)."""
    if not is_enabled():
        return
    text = report()
    if "no samples" in text:
        return
    print(f"\nReservation accuracy (leftover at exit):\n{text}\n", flush=True)


atexit.register(_atexit_report)
