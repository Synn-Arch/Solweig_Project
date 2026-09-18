"""Tests for the realtime latency-stage telemetry module.

Failing-first packet: written before ``solweig_gpu/server/realtime/telemetry.py``
exists. Percentile semantics are pinned to nearest-rank (NIST): for ``n``
samples the p-th percentile is the ``ceil(p/100 * n)``-th smallest sample,
so ``1..100`` yields p50=50, p95=95, p99=99.
"""

from __future__ import annotations

import threading
import time

import pytest

from solweig_gpu.server.realtime.telemetry import (
    DEFAULT_HISTOGRAM_CAPACITY,
    METRIC_ACK_MS,
    METRIC_EPOCHS_CLOSED_TOTAL,
    METRIC_EPOCH_WAIT_MS,
    METRIC_EXACT_COMPUTE_MS,
    METRIC_EXACT_QUEUE_MS,
    METRIC_FAST_COMPUTE_MS,
    METRIC_FAST_TOTAL_MS,
    METRIC_HOST_LOAD,
    METRIC_CORRECTION_MS,
    METRIC_OPERATIONS_ACCEPTED_TOTAL,
    METRIC_OPERATIONS_DUPLICATED_TOTAL,
    METRIC_PUBLISH_MS,
    METRIC_REDUCE_MS,
    METRIC_RESULT_CLASS_TOTAL,
    METRIC_RSS_BYTES,
    METRIC_STALE_PUBLISH_REJECTED_TOTAL,
    TelemetryRegistry,
    register_standard_metrics,
)


class FakeClock:
    """Deterministic monotonic clock for latency-math tests."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.now_ns = start_ns

    def __call__(self) -> int:
        return self.now_ns

    def advance_ms(self, ms: float) -> None:
        self.now_ns += int(ms * 1_000_000)


def make_registry(clock: FakeClock | None = None) -> TelemetryRegistry:
    if clock is None:
        return TelemetryRegistry()
    return TelemetryRegistry(time_ns=clock)


# -- counter ----------------------------------------------------------------


def test_counter_basics_and_monotonic_guard() -> None:
    reg = make_registry()
    ops = reg.counter("solweig_rt_test_ops_total")
    ops.inc()
    ops.inc(3)
    assert ops.value() == 4
    assert reg.snapshot()["solweig_rt_test_ops_total"][""]["value"] == 4
    with pytest.raises(ValueError):
        ops.inc(-1)
    with pytest.raises(TypeError):
        ops.inc(1.5)  # type: ignore[arg-type]
    assert ops.value() == 4  # guard rejected without mutating


def test_counter_label_isolation() -> None:
    reg = make_registry()
    classes = reg.counter("solweig_rt_result_class_total")
    for _ in range(3):
        classes.inc(labels={"class": "fast_exact"})
    for _ in range(5):
        classes.inc(labels={"class": "visual_pending"})
    snap = reg.snapshot()["solweig_rt_result_class_total"]
    assert snap["class=fast_exact"]["value"] == 3
    assert snap["class=visual_pending"]["value"] == 5
    assert classes.value(labels={"class": "fast_exact"}) == 3


# -- histogram --------------------------------------------------------------


def test_histogram_basics() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_ack_ms")
    lat.observe(12.5)
    lat.observe(13.0)
    stats = reg.snapshot()["solweig_rt_ack_ms"][""]
    assert stats["count"] == 2
    assert stats["last"] == 13.0
    assert stats["max"] == 13.0
    assert stats["mean"] == pytest.approx(12.75)
    assert stats["samples_retained"] == 2


def test_histogram_ring_overwrite_at_capacity() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_ring_ms", capacity=4)
    for value in range(1, 7):  # 1..6, ring keeps last 4 -> [3, 4, 5, 6]
        lat.observe(float(value))
    stats = reg.snapshot()["solweig_rt_ring_ms"][""]
    assert stats["count"] == 6  # total observations ever
    assert stats["samples_retained"] == 4
    assert stats["last"] == 6.0
    assert stats["max"] == 6.0
    assert stats["mean"] == pytest.approx(4.5)  # over retained [3, 4, 5, 6]
    # nearest-rank on sorted [3, 4, 5, 6]: p50 -> 2nd -> 4
    assert stats["p50"] == 4.0
    assert stats["p95"] == 6.0
    assert stats["p99"] == 6.0


def test_histogram_percentiles_pinned_nearest_rank() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_pinned_ms", capacity=DEFAULT_HISTOGRAM_CAPACITY)
    for value in range(1, 101):  # 1..100
        lat.observe(float(value))
    stats = reg.snapshot()["solweig_rt_pinned_ms"][""]
    # nearest-rank: rank = ceil(p/100 * n); p50 -> 50th -> 50, p95 -> 95, p99 -> 99
    assert stats["p50"] == 50.0
    assert stats["p95"] == 95.0
    assert stats["p99"] == 99.0
    assert stats["max"] == 100.0
    assert stats["mean"] == pytest.approx(50.5)
    assert stats["samples_retained"] == 100

    small = reg.histogram("solweig_rt_small_ms")
    for value in (1.0, 2.0, 3.0, 4.0):
        small.observe(value)
    small_stats = reg.snapshot()["solweig_rt_small_ms"][""]
    # n=4: p50 -> ceil(2.0)=2nd -> 2; p95 -> ceil(3.8)=4th -> 4; p99 -> 4
    assert small_stats["p50"] == 2.0
    assert small_stats["p95"] == 4.0
    assert small_stats["p99"] == 4.0

    one = reg.histogram("solweig_rt_one_ms")
    one.observe(7.0)
    one_stats = reg.snapshot()["solweig_rt_one_ms"][""]
    assert (one_stats["p50"], one_stats["p95"], one_stats["p99"]) == (7.0, 7.0, 7.0)


def test_histogram_label_isolation() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_label_ms")
    lat.observe(1.0, labels={"workspace": "a"})
    lat.observe(100.0, labels={"workspace": "b"})
    lat.observe(2.0, labels={"workspace": "a"})
    snap = reg.snapshot()["solweig_rt_label_ms"]
    assert snap["workspace=a"]["count"] == 2
    assert snap["workspace=a"]["max"] == 2.0
    assert snap["workspace=b"]["count"] == 1
    assert snap["workspace=b"]["max"] == 100.0
    assert set(snap) == {"workspace=a", "workspace=b"}


def test_histogram_rejects_nonfinite() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_finite_ms")
    with pytest.raises(ValueError):
        lat.observe(float("nan"))
    with pytest.raises(ValueError):
        lat.observe(float("inf"))
    # validation precedes any state change: no series, count stays 0
    assert lat.count() == 0
    assert "solweig_rt_finite_ms" not in reg.snapshot()


# -- gauge ------------------------------------------------------------------


def test_gauge_last_value_wins() -> None:
    reg = make_registry()
    rss = reg.gauge("solweig_rt_rss_bytes")
    assert rss.value() is None
    rss.set(1024.0)
    rss.set(2048.0)
    assert rss.value() == 2048.0
    assert reg.snapshot()["solweig_rt_rss_bytes"][""]["value"] == 2048.0
    load = reg.gauge("solweig_rt_host_load")
    load.set(0.42, labels={"host": "primary"})
    assert load.value(labels={"host": "primary"}) == 0.42


# -- registry get_or_create ---------------------------------------------------


def test_get_or_create_idempotent_and_kind_conflict() -> None:
    reg = make_registry()
    first = reg.counter("solweig_rt_same_total")
    second = reg.counter("solweig_rt_same_total")
    first.inc(2)
    assert second.value() == 2  # same underlying metric
    reg.histogram("solweig_rt_same_ms")  # distinct name is fine
    with pytest.raises(ValueError):
        reg.counter("solweig_rt_same_ms")  # kind conflict
    with pytest.raises(ValueError):
        reg.gauge("solweig_rt_same_total")
    assert reg.metric_names() >= {
        "solweig_rt_same_total",
        "solweig_rt_same_ms",
    }


# -- clock discipline ---------------------------------------------------------


def test_fake_clock_injection_and_stage_timer() -> None:
    clock = FakeClock()
    reg = make_registry(clock)
    lat = reg.histogram("solweig_rt_stage_ms")
    with reg.timer("solweig_rt_stage_ms"):
        clock.advance_ms(2.5)
    assert lat.count() == 1
    stats = reg.snapshot()["solweig_rt_stage_ms"][""]
    assert stats["last"] == pytest.approx(2.5)

    clock.advance_ms(10.0)
    with reg.timer("solweig_rt_stage_ms", labels={"lane": "fast"}):
        clock.advance_ms(7.25)
    snap = reg.snapshot()["solweig_rt_stage_ms"]
    assert snap[""]["count"] == 1
    assert snap["lane=fast"]["last"] == pytest.approx(7.25)


def test_timer_observes_even_when_body_raises() -> None:
    reg = make_registry()
    with pytest.raises(RuntimeError):
        with reg.timer("solweig_rt_boom_ms"):
            raise RuntimeError("boom")
    assert reg.snapshot()["solweig_rt_boom_ms"][""]["count"] == 1


# -- snapshot shape ------------------------------------------------------------


def test_snapshot_shape() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_ack_ms")
    lat.observe(5.0)
    lat.observe(6.0, labels={"workspace": "w1"})
    ctr = reg.counter("solweig_rt_epochs_closed_total")
    ctr.inc()
    snap = reg.snapshot()
    assert set(snap["solweig_rt_ack_ms"][""]) == {
        "count",
        "p50",
        "p95",
        "p99",
        "max",
        "mean",
        "last",
        "samples_retained",
    }
    assert "value" in snap["solweig_rt_epochs_closed_total"][""]


# -- render_text ----------------------------------------------------------------


def test_render_text_stability_and_format() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_ack_ms", doc="ack latency")
    lat.observe(10.0)
    lat.observe(20.0)
    lat.observe(30.0, labels={"class": "fast_exact"})
    ctr = reg.counter("solweig_rt_operations_accepted_total", doc="accepted ops")
    ctr.inc(4)
    gauge = reg.gauge("solweig_rt_host_load", doc="host load")
    gauge.set(1.25)

    first = reg.render_text()
    second = reg.render_text()
    assert first == second  # identical bytes for identical data

    lines = first.splitlines()
    assert "# TYPE solweig_rt_ack_ms gauge" in lines
    assert "# TYPE solweig_rt_operations_accepted_total counter" in lines
    assert "# TYPE solweig_rt_host_load gauge" in lines
    assert "solweig_rt_operations_accepted_total 4" in lines
    assert "solweig_rt_host_load 1.25" in lines
    assert 'solweig_rt_ack_ms_count{class="fast_exact"} 1' in lines
    assert 'solweig_rt_ack_ms{class="fast_exact",quantile="0.5"} 30.0' in lines
    # nearest-rank on [10, 20]: n=2, p50 -> rank ceil(1.0)=1 -> 10.0
    assert 'solweig_rt_ack_ms{quantile="0.5"} 10.0' in lines
    # metric blocks are emitted in sorted name order
    names = [ln.split()[2] for ln in lines if ln.startswith("# TYPE")]
    assert names == sorted(names)


# -- thread safety ----------------------------------------------------------------


def test_thread_safety_smoke_exact_totals() -> None:
    reg = make_registry()
    ctr = reg.counter("solweig_rt_threaded_total")
    lat = reg.histogram("solweig_rt_threaded_ms")
    errors: list[Exception] = []
    threads = 8
    per_thread = 10_000

    def worker(worker_id: int) -> None:
        try:
            for i in range(per_thread):
                ctr.inc()
                lat.observe(float(i))
        except Exception as exc:  # pragma: no cover - surfaced via assert
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert ctr.value() == threads * per_thread
    stats = reg.snapshot()["solweig_rt_threaded_ms"][""]
    assert stats["count"] == threads * per_thread
    assert stats["samples_retained"] == DEFAULT_HISTOGRAM_CAPACITY


# -- O(1) record guard ----------------------------------------------------------------


def test_record_path_is_constant_time() -> None:
    reg = make_registry()
    lat = reg.histogram("solweig_rt_speed_ms")
    records = 10_000
    start = time.perf_counter()
    for i in range(records):
        lat.observe(float(i))
    elapsed = time.perf_counter() - start
    assert elapsed < 0.050, f"{records} records took {elapsed * 1000:.1f} ms"
    assert lat.count() == records


# -- standard metric set ----------------------------------------------------------------


def test_register_standard_metrics_names_and_handles() -> None:
    reg = make_registry()
    metrics = register_standard_metrics(reg)
    expected = {
        METRIC_ACK_MS,
        METRIC_EPOCH_WAIT_MS,
        METRIC_REDUCE_MS,
        METRIC_FAST_COMPUTE_MS,
        METRIC_PUBLISH_MS,
        METRIC_FAST_TOTAL_MS,
        METRIC_EXACT_QUEUE_MS,
        METRIC_EXACT_COMPUTE_MS,
        METRIC_CORRECTION_MS,
        METRIC_RSS_BYTES,
        METRIC_HOST_LOAD,
        METRIC_RESULT_CLASS_TOTAL,
        METRIC_OPERATIONS_ACCEPTED_TOTAL,
        METRIC_OPERATIONS_DUPLICATED_TOTAL,
        METRIC_EPOCHS_CLOSED_TOTAL,
        METRIC_STALE_PUBLISH_REJECTED_TOTAL,
    }
    assert len(expected) == 16
    assert expected <= set(reg.metric_names())

    # constants carry the solweig_rt_ convention
    for name in expected:
        assert name.startswith("solweig_rt_")

    # handles are wired to the registered names
    metrics.ack_ms.observe(1.0)
    metrics.operations_accepted_total.inc()
    metrics.epochs_closed_total.inc(2)
    metrics.result_class_total.inc(labels={"class": "fast_exact"})
    metrics.rss_bytes.set(512.0)
    metrics.host_load.set(0.5)
    snap = reg.snapshot()
    assert snap[METRIC_ACK_MS][""]["count"] == 1
    assert snap[METRIC_OPERATIONS_ACCEPTED_TOTAL][""]["value"] == 1
    assert snap[METRIC_EPOCHS_CLOSED_TOTAL][""]["value"] == 2
    assert snap[METRIC_RESULT_CLASS_TOTAL]["class=fast_exact"]["value"] == 1
    assert snap[METRIC_RSS_BYTES][""]["value"] == 512.0
    assert snap[METRIC_HOST_LOAD][""]["value"] == 0.5


def test_register_standard_metrics_defaults_to_shared_singleton() -> None:
    from solweig_gpu.server.realtime.telemetry import default_registry

    metrics = register_standard_metrics()
    try:
        assert METRIC_ACK_MS in default_registry().metric_names()
        metrics.publish_ms.observe(3.0)
        assert (
            default_registry().snapshot()[METRIC_PUBLISH_MS][""]["count"] == 1
        )
    finally:  # keep the module singleton clean for other tests
        default_registry().reset()


def test_standard_metrics_render_prometheus_lines() -> None:
    reg = make_registry()
    metrics = register_standard_metrics(reg)
    metrics.ack_ms.observe(12.0)
    text = reg.render_text()
    assert "# TYPE solweig_rt_ack_ms gauge" in text
    assert "solweig_rt_ack_ms_last 12.0" in text
    assert "solweig_rt_ack_ms{quantile=\"0.5\"} 12.0" in text
