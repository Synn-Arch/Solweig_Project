"""Latency-stage telemetry for the SOLWEIG realtime collaboration lanes.

Pure, dependency-free (stdlib only), thread-safe counters/histograms/gauges
intended for the R2 epoch scheduler and the realtime routes. This module
never imports the store (no shared RLock) and never samples the wall clock:
all latency math flows through the monotonic clock callable injected at
registry construction (default ``time.monotonic_ns``).

Design notes
------------
* **Record path is O(1) and allocation-light.** Every (metric, label-set)
  pair owns a preallocated float ring buffer; recording stores one float at
  the write index and overwrites the oldest sample once the ring wraps.
  ``record`` never sorts, never copies, never renders.
* **Snapshots may allocate.** ``snapshot()`` and ``render_text()`` copy and
  sort the retained ring — they run on the ops/scrape path, not the hot path.
* **Percentiles are nearest-rank (NIST).** For ``n`` samples the ``p``-th
  percentile is the ``ceil(p/100 * n)``-th smallest sample (1-based). For
  ``1..100`` that pins p50=50, p95=95, p99=99. All statistics (mean, max,
  quantiles) are computed over the *retained* ring; ``count`` is the total
  number of observations ever recorded.
* **Gauges are a dedicated primitive** (not a degenerate histogram): RSS and
  host load are point-in-time readings, so a histogram ring would be pure
  overhead. ``Gauge.set`` keeps only the last value per label-set.
* **Rendering convention.** Rings are quantile summaries, not cumulative
  Prometheus histograms, so ``render_text()`` emits each histogram metric as
  ``# TYPE <name> gauge`` with explicit ``quantile="0.5|0.95|0.99"`` samples
  plus ``<name>_count`` / ``<name>_sum`` / ``<name>_last`` / ``<name>_max``
  companion gauges (the solweig convention). Counters render as
  ``# TYPE <name> counter``, gauges as ``# TYPE <name> gauge``. Output is
  fully deterministic: metric names sorted, label-sets sorted, fields in a
  fixed order — identical data renders identical bytes.
* **Label keys** are canonicalised as sorted ``key=value`` pairs joined by
  commas (e.g. ``class=fast_exact``); unlabelled series use the empty
  string. Label values are internal controlled enums and are not escaped.

This package directory is a PEP 420 namespace package on purpose: parallel
realtime modules (reducer, ops) drop files beside this one without every
wave having to own a shared ``__init__.py``.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping

__all__ = [
    "DEFAULT_HISTOGRAM_CAPACITY",
    "Counter",
    "Gauge",
    "Histogram",
    "StandardMetrics",
    "TelemetryRegistry",
    "METRIC_ACK_MS",
    "METRIC_EPOCH_WAIT_MS",
    "METRIC_EPOCH_TICK_LATENESS_MS",
    "METRIC_REDUCE_MS",
    "METRIC_FAST_COMPUTE_MS",
    "METRIC_PUBLISH_MS",
    "METRIC_FAST_TOTAL_MS",
    "METRIC_EXACT_QUEUE_MS",
    "METRIC_EXACT_COMPUTE_MS",
    "METRIC_CORRECTION_MS",
    "METRIC_RSS_BYTES",
    "METRIC_HOST_LOAD",
    "METRIC_RESULT_CLASS_TOTAL",
    "METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL",
    "METRIC_OPERATIONS_ACCEPTED_TOTAL",
    "METRIC_OPERATIONS_DUPLICATED_TOTAL",
    "METRIC_EPOCHS_CLOSED_TOTAL",
    "METRIC_STALE_PUBLISH_REJECTED_TOTAL",
    "default_registry",
    "register_standard_metrics",
]

DEFAULT_HISTOGRAM_CAPACITY = 4096

# -- standard metric names (R0 telemetry schema, design decision 7) ---------
# Stage budgeting per epoch_scheduler.md: ack, epoch wait, reduce, fast
# compute, publish, and the headline fast-total; plus exact-lane queue and
# compute, and exact->fast correction lag.

METRIC_ACK_MS = "solweig_rt_ack_ms"
METRIC_EPOCH_WAIT_MS = "solweig_rt_epoch_wait_ms"
#: R8b: how much later than one tick interval the scheduler's wake loop
#: actually fired (>= 0). Direct evidence for tail attribution: an
#: epoch's wait beyond its [W, 2W) floor is bounded by tick lateness.
METRIC_EPOCH_TICK_LATENESS_MS = "solweig_rt_epoch_tick_lateness_ms"
METRIC_REDUCE_MS = "solweig_rt_reduce_ms"
METRIC_FAST_COMPUTE_MS = "solweig_rt_fast_compute_ms"
METRIC_PUBLISH_MS = "solweig_rt_publish_ms"
METRIC_FAST_TOTAL_MS = "solweig_rt_fast_total_ms"
METRIC_EXACT_QUEUE_MS = "solweig_rt_exact_queue_ms"
METRIC_EXACT_COMPUTE_MS = "solweig_rt_exact_compute_ms"
METRIC_CORRECTION_MS = "solweig_rt_correction_ms"

# Gauges (point-in-time readings sampled from the ops thread).
METRIC_RSS_BYTES = "solweig_rt_rss_bytes"
METRIC_HOST_LOAD = "solweig_rt_host_load"

# Counters.
METRIC_RESULT_CLASS_TOTAL = "solweig_rt_result_class_total"
#: R7: fast_qualified results structurally downgraded to visual_pending
#: (missing/malformed qualifier), labeled by the bounded defect vocabulary
#: (``qualification.QUALIFIER_DEFECTS``) — "never reduce precision silently".
METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL = "solweig_rt_fast_qualified_downgraded_total"
METRIC_OPERATIONS_ACCEPTED_TOTAL = "solweig_rt_operations_accepted_total"
METRIC_OPERATIONS_DUPLICATED_TOTAL = "solweig_rt_operations_duplicated_total"
METRIC_EPOCHS_CLOSED_TOTAL = "solweig_rt_epochs_closed_total"
METRIC_STALE_PUBLISH_REJECTED_TOTAL = "solweig_rt_stale_publish_rejected_total"


def _labels_key(labels: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Canonical hashable key for a label set (sorted pairs; () when none)."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def _labels_text(key: tuple[tuple[str, str], ...]) -> str:
    """Render a label key as the canonical ``a=1,b=2`` string."""
    return ",".join(f"{name}={value}" for name, value in key)


def _prom_labels(
    key: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None
) -> str:
    """Prometheus label selector text; ``extra`` is appended last."""
    pairs = key if extra is None else key + (extra,)
    if not pairs:
        return ""
    return "{" + ",".join(f'{name}="{value}"' for name, value in pairs) + "}"


def _fmt(value: float) -> str:
    """Stable float rendering (shortest round-trip)."""
    return repr(float(value))


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on an ascending-sorted list.

    rank = ceil(fraction * n), clamped to [1, n], 1-based.
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("percentile of empty sample set")
    rank = max(1, math.ceil(fraction * n))
    return sorted_values[rank - 1]


class _Ring:
    """Fixed-capacity float ring buffer. ``append`` is O(1), no allocation."""

    __slots__ = ("_buf", "_head", "_count", "_last")

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buf: list[float] = [0.0] * capacity
        self._head = 0  # next write index
        self._count = 0  # total observations ever
        self._last = 0.0

    def append(self, value: float) -> None:
        buf = self._buf
        head = self._head
        buf[head] = value
        head += 1
        if head == len(buf):
            head = 0
        self._head = head
        self._count += 1
        self._last = value

    @property
    def count(self) -> int:
        return self._count

    @property
    def last(self) -> float:
        return self._last

    def retained(self) -> list[float]:
        """Retained samples, oldest first. Snapshot path only (allocates)."""
        count = self._count
        buf = self._buf
        if count <= len(buf):
            return buf[:count]
        head = self._head
        return buf[head:] + buf[:head]

    def stats(self) -> dict[str, float | int]:
        """Summary statistics over the retained window (allocates/sorts)."""
        samples = self.retained()
        ordered = sorted(samples)
        total = sum(samples)
        return {
            "count": self._count,
            "p50": _percentile(ordered, 0.50),
            "p95": _percentile(ordered, 0.95),
            "p99": _percentile(ordered, 0.99),
            "max": ordered[-1],
            "mean": total / len(samples),
            "last": self._last,
            "samples_retained": len(samples),
        }


class _Metric:
    """Shared state for one metric name: kind, doc, per-label-set series.

    ``lock`` guards the series map and every series mutation. It is a plain
    per-metric lock — never the store RLock (this module is store-agnostic).
    """

    __slots__ = ("name", "kind", "doc", "capacity", "lock", "series")

    def __init__(self, name: str, kind: str, doc: str, capacity: int) -> None:
        self.name = name
        self.kind = kind  # "counter" | "histogram" | "gauge"
        self.doc = doc
        self.capacity = capacity
        self.lock = threading.Lock()
        # counter: labels_key -> int; histogram: -> _Ring; gauge: -> float
        self.series: dict[tuple[tuple[str, str], ...], object] = {}

    def key_text(self, key: tuple[tuple[str, str], ...]) -> str:
        return _labels_text(key)


def _check_labels(labels: Mapping[str, str] | None) -> None:
    if labels is None:
        return
    for name, value in labels.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("label names and values must be strings")


class Counter:
    """Monotonic integer counter with optional label sets."""

    __slots__ = ("_metric",)

    def __init__(self, metric: _Metric) -> None:
        self._metric = metric

    def inc(self, value: int = 1, *, labels: Mapping[str, str] | None = None) -> None:
        """Increment by a non-negative int (counters only ever go up)."""
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("counter increments must be int")
        if value < 0:
            raise ValueError("counter increments must be non-negative")
        _check_labels(labels)
        key = _labels_key(labels)
        metric = self._metric
        with metric.lock:
            current = metric.series.get(key)
            if current is None:
                metric.series[key] = value
            else:
                metric.series[key] = current + value  # type: ignore[operator]

    def value(self, *, labels: Mapping[str, str] | None = None) -> int:
        key = _labels_key(labels)
        with self._metric.lock:
            return self._metric.series.get(key, 0)  # type: ignore[return-value]


class Histogram:
    """Raw-sample ring histogram with optional label sets."""

    __slots__ = ("_metric",)

    def __init__(self, metric: _Metric) -> None:
        self._metric = metric

    def observe(
        self, value: float, *, labels: Mapping[str, str] | None = None
    ) -> None:
        """Record one sample. O(1): one float store into a preallocated ring."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("histogram samples must be int or float")
        if not math.isfinite(value):
            raise ValueError("histogram samples must be finite")
        _check_labels(labels)
        key = _labels_key(labels)
        metric = self._metric
        with metric.lock:
            ring = metric.series.get(key)
            if ring is None:
                ring = _Ring(metric.capacity)
                metric.series[key] = ring
            ring.append(float(value))  # type: ignore[union-attr]

    def count(self, *, labels: Mapping[str, str] | None = None) -> int:
        key = _labels_key(labels)
        with self._metric.lock:
            ring = self._metric.series.get(key)
            return ring.count if ring is not None else 0  # type: ignore[union-attr]


class Gauge:
    """Point-in-time value with optional label sets (last set wins)."""

    __slots__ = ("_metric",)

    def __init__(self, metric: _Metric) -> None:
        self._metric = metric

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("gauge values must be int or float")
        if not math.isfinite(value):
            raise ValueError("gauge values must be finite")
        _check_labels(labels)
        key = _labels_key(labels)
        metric = self._metric
        with metric.lock:
            metric.series[key] = float(value)

    def value(self, *, labels: Mapping[str, str] | None = None) -> float | None:
        key = _labels_key(labels)
        with self._metric.lock:
            value = self._metric.series.get(key)
        return None if value is None else float(value)  # type: ignore[arg-type]


class _StageTimer:
    """Context manager observing elapsed ms into a histogram.

    Uses the registry's injected monotonic clock — the module never touches
    the wall clock itself. Always records, including on exception.
    """

    __slots__ = ("_histogram", "_time_ns", "_labels", "_start_ns")

    def __init__(
        self,
        histogram: Histogram,
        time_ns: Callable[[], int],
        labels: Mapping[str, str] | None,
    ) -> None:
        self._histogram = histogram
        self._time_ns = time_ns
        self._labels = labels
        self._start_ns = 0

    def __enter__(self) -> "_StageTimer":
        self._start_ns = self._time_ns()
        return self

    def __exit__(self, *exc_info: object) -> None:
        elapsed_ms = (self._time_ns() - self._start_ns) / 1_000_000.0
        self._histogram.observe(elapsed_ms, labels=self._labels)


class TelemetryRegistry:
    """Registry of counters/histograms/gauges with ``get_or_create`` handles.

    ``time_ns`` injects the monotonic clock used for latency math (tests
    inject fakes; production uses the ``time.monotonic_ns`` default).
    """

    def __init__(self, time_ns: Callable[[], int] = time.monotonic_ns) -> None:
        self._time_ns = time_ns
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()  # guards the metric map only

    # -- handle access (get_or_create) --------------------------------

    def _get_or_create(
        self, name: str, kind: str, doc: str, capacity: int
    ) -> _Metric:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = _Metric(name, kind, doc, capacity)
                self._metrics[name] = metric
                return metric
        if metric.kind != kind:
            raise ValueError(
                f"metric {name!r} already registered as {metric.kind}, "
                f"cannot re-register as {kind}"
            )
        return metric

    def counter(self, name: str, doc: str = "") -> Counter:
        return Counter(self._get_or_create(name, "counter", doc, 0))

    def histogram(
        self, name: str, capacity: int = DEFAULT_HISTOGRAM_CAPACITY, doc: str = ""
    ) -> Histogram:
        return Histogram(self._get_or_create(name, "histogram", doc, capacity))

    def gauge(self, name: str, doc: str = "") -> Gauge:
        return Gauge(self._get_or_create(name, "gauge", doc, 0))

    def timer(
        self, name: str, *, labels: Mapping[str, str] | None = None
    ) -> _StageTimer:
        """Time a block into histogram ``name`` (ms, injected clock)."""
        return _StageTimer(self.histogram(name), self._time_ns, labels)

    def metric_names(self) -> set[str]:
        with self._lock:
            return set(self._metrics)

    def reset(self) -> None:
        """Drop all metrics (test/ops helper). Not a hot-path operation."""
        with self._lock:
            self._metrics.clear()

    # -- snapshot / rendering (allowed to allocate) --------------------

    def snapshot(self) -> dict[str, dict[str, dict[str, float | int]]]:
        """Point-in-time snapshot.

        Histogram metric ``m`` contributes, per label-set key::

            {m: {key: {count, p50, p95, p99, max, mean, last,
                       samples_retained}}}

        Counters/gauges contribute ``{m: {key: {"value": v}}}``.
        """
        result: dict[str, dict[str, dict[str, float | int]]] = {}
        with self._lock:
            metrics = list(self._metrics.values())
        for metric in metrics:
            with metric.lock:
                if metric.kind == "histogram":
                    series_view = {
                        _labels_text(key): ring.stats()  # type: ignore[union-attr]
                        for key, ring in metric.series.items()
                    }
                else:
                    series_view = {
                        _labels_text(key): {"value": value}
                        for key, value in metric.series.items()
                    }
            if series_view:
                result[metric.name] = series_view
        return result

    def render_text(self) -> str:
        """Prometheus exposition text; deterministic for identical data."""
        lines: list[str] = []
        with self._lock:
            metrics = sorted(self._metrics.values(), key=lambda m: m.name)
        for metric in metrics:
            with metric.lock:
                items = sorted(
                    ((key, value) for key, value in metric.series.items()),
                    key=lambda item: _labels_text(item[0]),
                )
                kind = metric.kind
                name = metric.name
                doc = metric.doc
            if not items:
                continue
            if doc:
                lines.append(f"# HELP {name} {doc}")
            if kind == "histogram":
                # solweig convention: quantile gauges + companion stats.
                lines.append(f"# TYPE {name} gauge")
                for key, value in items:
                    ring: _Ring = value  # type: ignore[assignment]
                    stats = ring.stats()
                    selector = _prom_labels(key)
                    lines.append(f"{name}_count{selector} {stats['count']}")
                    mean = stats["mean"]
                    count = stats["count"]
                    lines.append(
                        f"{name}_sum{selector} {_fmt(mean * count)}"
                    )
                    lines.append(f"{name}_last{selector} {_fmt(stats['last'])}")
                    lines.append(f"{name}_max{selector} {_fmt(stats['max'])}")
                    lines.append(
                        f"{name}{_prom_labels(key, ('quantile', '0.5'))}"
                        f" {_fmt(stats['p50'])}"
                    )
                    lines.append(
                        f"{name}{_prom_labels(key, ('quantile', '0.95'))}"
                        f" {_fmt(stats['p95'])}"
                    )
                    lines.append(
                        f"{name}{_prom_labels(key, ('quantile', '0.99'))}"
                        f" {_fmt(stats['p99'])}"
                    )
            elif kind == "counter":
                lines.append(f"# TYPE {name} counter")
                for key, value in items:
                    lines.append(f"{name}{_prom_labels(key)} {value}")
            else:  # gauge
                lines.append(f"# TYPE {name} gauge")
                for key, value in items:
                    lines.append(f"{name}{_prom_labels(key)} {_fmt(value)}")  # type: ignore[arg-type]
            lines.append("")
        return "\n".join(lines).rstrip("\n") + "\n"


class StandardMetrics:
    """Handles for the standard solweig_rt_* metric set."""

    __slots__ = (
        "ack_ms",
        "epoch_wait_ms",
        "epoch_tick_lateness_ms",
        "reduce_ms",
        "fast_compute_ms",
        "publish_ms",
        "fast_total_ms",
        "exact_queue_ms",
        "exact_compute_ms",
        "correction_ms",
        "rss_bytes",
        "host_load",
        "result_class_total",
        "fast_qualified_downgraded_total",
        "operations_accepted_total",
        "operations_duplicated_total",
        "epochs_closed_total",
        "stale_publish_rejected_total",
    )

    def __init__(self, registry: TelemetryRegistry) -> None:
        self.ack_ms = registry.histogram(
            METRIC_ACK_MS,
            doc="t_accept - t_receive: acknowledge latency in ms",
        )
        self.epoch_wait_ms = registry.histogram(
            METRIC_EPOCH_WAIT_MS,
            doc="t_epoch_close - t_accept: wait to epoch close in ms",
        )
        self.epoch_tick_lateness_ms = registry.histogram(
            METRIC_EPOCH_TICK_LATENESS_MS,
            doc="scheduler tick wake lateness beyond one interval in ms "
            "(R8b tail-attribution evidence)",
        )
        self.reduce_ms = registry.histogram(
            METRIC_REDUCE_MS,
            doc="epoch reduction + dependency planning in ms",
        )
        self.fast_compute_ms = registry.histogram(
            METRIC_FAST_COMPUTE_MS,
            doc="fast-lane compute per epoch in ms",
        )
        self.publish_ms = registry.histogram(
            METRIC_PUBLISH_MS,
            doc="encode + publish + broadcast of the fast revision in ms",
        )
        self.fast_total_ms = registry.histogram(
            METRIC_FAST_TOTAL_MS,
            doc="t_fast_publish - t_accept: headline fast latency in ms",
        )
        self.exact_queue_ms = registry.histogram(
            METRIC_EXACT_QUEUE_MS,
            doc="exact-lane target wait before compute starts in ms",
        )
        self.exact_compute_ms = registry.histogram(
            METRIC_EXACT_COMPUTE_MS,
            doc="exact SOLWEIG reconciliation compute in ms",
        )
        self.correction_ms = registry.histogram(
            METRIC_CORRECTION_MS,
            doc="exact->fast correction publish lag in ms",
        )
        self.rss_bytes = registry.gauge(
            METRIC_RSS_BYTES,
            doc="resident set size of the server process in bytes",
        )
        self.host_load = registry.gauge(
            METRIC_HOST_LOAD,
            doc="host load average sampled by the ops thread",
        )
        self.result_class_total = registry.counter(
            METRIC_RESULT_CLASS_TOTAL,
            doc="published fast results by dependency class",
        )
        self.fast_qualified_downgraded_total = registry.counter(
            METRIC_FAST_QUALIFIED_DOWNGRADED_TOTAL,
            doc="fast_qualified results downgraded for qualifier defects, "
            "by defect id (never reduce precision silently)",
        )
        self.operations_accepted_total = registry.counter(
            METRIC_OPERATIONS_ACCEPTED_TOTAL,
            doc="operations accepted into the durable audit",
        )
        self.operations_duplicated_total = registry.counter(
            METRIC_OPERATIONS_DUPLICATED_TOTAL,
            doc="idempotent duplicate operations collapsed to one effect",
        )
        self.epochs_closed_total = registry.counter(
            METRIC_EPOCHS_CLOSED_TOTAL,
            doc="workspace epochs closed by the scheduler",
        )
        self.stale_publish_rejected_total = registry.counter(
            METRIC_STALE_PUBLISH_REJECTED_TOTAL,
            doc="stale publication attempts rejected by the revision fence",
        )


_DEFAULT_REGISTRY = TelemetryRegistry()


def default_registry() -> TelemetryRegistry:
    """Process-wide shared registry (used by the /metrics route later)."""
    return _DEFAULT_REGISTRY


def register_standard_metrics(
    registry: TelemetryRegistry | None = None,
) -> StandardMetrics:
    """Register the standard solweig_rt_* set; returns ready handles."""
    return StandardMetrics(registry if registry is not None else default_registry())
