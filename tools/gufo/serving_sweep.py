#!/usr/bin/env python3
"""Joint `max_num_seqs` / `max_num_batched_tokens` sweep with knee detection.

The PRD's primary objective is aggregate decode throughput, which grows almost
linearly with the admitted batch until a memory or compute limit flattens it.
This tool measures that curve for one model and reports the knee point, so a
tuning run never has to guess a batch size. The knee is computed on delivered
total tokens per second: the summed per-request decode rate is reported too, but
it counts each request's own decode phase and therefore overstates concurrency
when requests wait behind a narrower execution wave. Delivered throughput cannot
do that, so it decides where adding slots stops paying.

Two axes are swept together:

* `sessions` (`--sessions`) sets how many requests may be resident, i.e. the
  batch width the scheduler admits.
* `prefill` (`--prefill-chunk`) is the token budget a scheduled prefill chunk
  may spend between active decode rounds, i.e. the batched-token budget that
  competes with the batch for the same iteration.

Both are server options, so every measured point is a fresh process. The
sessions axis is therefore swept inner and stops at the knee, which keeps the
number of model loads proportional to the prefill axis rather than to the
product of both axes.

The measurement itself is the canonical OpenAI-compatible serving benchmark
(`gufo.serving_bench`), which keeps stage throughput, latency and aggregate
concurrency throughput separate.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCHEMA = "gufo.serving-sweep.v1"

#: Throughput gain a further batch step must deliver to stay on the rising part
#: of the curve. Below it the batch is saturated: the extra slots cost memory
#: and latency without adding tokens, so the knee is the previous point.
DEFAULT_MIN_GAIN_RATIO = 0.03

#: Seconds to let the driver release a finished point's unified-memory mappings
#: after the device already reports them free.
SETTLE_SECONDS = 20.0
#: A point's weights live in GPU-visible unified memory. The process exits
#: before the driver finishes tearing those mappings down, so the next point's
#: load can fail with std::bad_alloc on an otherwise idle host. Wait for the
#: device to report itself free, bounded by this many seconds.
DEVICE_RELEASE_TIMEOUT_SECONDS = 120.0
#: GTT below this is "the previous point is gone", not a partially freed model.
DEVICE_IDLE_BYTES = 1 << 30


def _device_used_bytes() -> int | None:
    """GPU-visible memory in use, or None when the host has no DRM sysfs."""
    for path in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_used"):
        try:
            return int(Path(path).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
    return None


def _wait_device_idle(timeout: float = DEVICE_RELEASE_TIMEOUT_SECONDS) -> None:
    """Block until the previous point's weights left GPU-visible memory."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = _device_used_bytes()
        if used is None or used < DEVICE_IDLE_BYTES:
            return
        time.sleep(0.5)


class SweepError(RuntimeError):
    """The sweep could not produce a measured knee."""


def parse_axis(text: str) -> list[int]:
    """Parse `1,2,4,8` or `1:128:2` into an ascending unique positive axis."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("axis must not be empty")
    values: list[int] = []
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            raise ValueError(f"invalid axis value: {part!r}")
        if ":" in item:
            bounds = item.split(":")
            if len(bounds) not in (2, 3):
                raise ValueError(f"invalid axis range: {item!r}")
            start, stop = (int(value) for value in bounds[:2])
            step = int(bounds[2]) if len(bounds) == 3 else 1
            if step <= 0 or stop < start:
                raise ValueError(f"invalid axis range: {item!r}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    if any(value <= 0 for value in values):
        raise ValueError("axis values must be positive")
    return sorted(set(values))


@dataclass(frozen=True)
class Measurement:
    """One measured (sessions, prefill) point."""

    sessions: int
    prefill_tokens: int
    total_tokens_per_second: float
    output_tokens_per_second: float
    decode_tokens_per_second: float | None = None
    prefill_tokens_per_second: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "sessions": self.sessions,
            "prefillTokens": self.prefill_tokens,
            "totalTokensPerSecond": self.total_tokens_per_second,
            "outputTokensPerSecond": self.output_tokens_per_second,
            "decodeTokensPerSecond": self.decode_tokens_per_second,
            "prefillTokensPerSecond": self.prefill_tokens_per_second,
        }


@dataclass(frozen=True)
class Knee:
    """Where the batch curve stops paying for another slot."""

    knee_sessions: int
    knee_total_tokens_per_second: float
    peak_sessions: int
    peak_total_tokens_per_second: float
    saturated: bool
    monotone: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kneeSessions": self.knee_sessions,
            "kneeTotalTokensPerSecond": self.knee_total_tokens_per_second,
            "peakSessions": self.peak_sessions,
            "peakTotalTokensPerSecond": self.peak_total_tokens_per_second,
            "saturated": self.saturated,
            "monotone": self.monotone,
            "reason": self.reason,
        }


def detect_knee(
    points: Sequence[Measurement], min_gain_ratio: float = DEFAULT_MIN_GAIN_RATIO
) -> Knee:
    """Locate the knee of an ascending-sessions throughput curve.

    Rising, saturated and the never-saturated case are reported apart, because
    only the last one means the axis was too short. A batch step that yields
    less aggregate throughput than its predecessor is the saturation point: the
    curve rose monotonically up to the peak before it, so the peak is the knee
    and the decline is recorded rather than smoothed away.
    """
    if not points:
        raise ValueError("knee detection needs at least one measured point")
    if min_gain_ratio < 0.0:
        raise ValueError("min_gain_ratio must not be negative")
    ordered = sorted(points, key=lambda point: point.sessions)
    knee = ordered[0]
    reason = "single measured point"
    for previous, current in zip(ordered, ordered[1:]):
        if current.total_tokens_per_second < previous.total_tokens_per_second:
            # A wider batch that yields less aggregate throughput is past the
            # knee: the peak just before it is where another slot stopped
            # paying. The curve rose monotonically up to that knee, which is
            # what the acceptance criterion requires; the decline is the
            # saturation signal, not noise to smooth away.
            decline = (
                current.total_tokens_per_second
                / previous.total_tokens_per_second
                - 1.0
            )
            return Knee(
                knee_sessions=previous.sessions,
                knee_total_tokens_per_second=previous.total_tokens_per_second,
                peak_sessions=previous.sessions,
                peak_total_tokens_per_second=previous.total_tokens_per_second,
                saturated=True,
                monotone=True,
                reason=(
                    f"batch step {previous.sessions}->{current.sessions} fell "
                    f"{-decline * 100.0:.1f}%"
                ),
            )
        gain = (
            current.total_tokens_per_second / previous.total_tokens_per_second - 1.0
            if previous.total_tokens_per_second > 0.0
            else 0.0
        )
        if gain < min_gain_ratio:
            reason = (
                f"batch step {previous.sessions}->{current.sessions} gained "
                f"{gain * 100.0:.1f}%"
            )
            return Knee(
                knee_sessions=knee.sessions,
                knee_total_tokens_per_second=knee.total_tokens_per_second,
                peak_sessions=max(
                    ordered, key=lambda point: point.total_tokens_per_second
                ).sessions,
                peak_total_tokens_per_second=max(
                    point.total_tokens_per_second for point in ordered
                ),
                saturated=True,
                monotone=True,
                reason=reason,
            )
        knee = current
    peak = max(ordered, key=lambda point: point.total_tokens_per_second)
    return Knee(
        knee_sessions=knee.sessions,
        knee_total_tokens_per_second=knee.total_tokens_per_second,
        peak_sessions=peak.sessions,
        peak_total_tokens_per_second=peak.total_tokens_per_second,
        saturated=False,
        monotone=True,
        reason="still rising at the largest session count",
    )


@dataclass
class AxisResult:
    """The sessions curve measured under one prefill budget."""

    prefill_tokens: int
    points: list[Measurement] = field(default_factory=list)
    knee: Knee | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "prefillTokens": self.prefill_tokens,
            "points": [point.as_dict() for point in self.points],
            "knee": self.knee.as_dict() if self.knee else None,
        }


@dataclass
class SweepResult:
    model: str
    axes: dict[str, list[int]]
    min_gain_ratio: float
    curves: list[AxisResult]
    best: Knee | None
    best_prefill_tokens: int | None
    violations: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "sweepSchema": SWEEP_SCHEMA,
            "artifactType": "servingSweep",
            "model": {"id": self.model},
            "axes": self.axes,
            "thresholds": {"minGainRatio": self.min_gain_ratio},
            "curves": [curve.as_dict() for curve in self.curves],
            "result": {
                "kneeSessions": self.best.knee_sessions if self.best else None,
                "kneePrefillTokens": self.best_prefill_tokens,
                "kneeTotalTokensPerSecond": (
                    self.best.knee_total_tokens_per_second if self.best else None
                ),
                "saturated": self.best.saturated if self.best else False,
            },
            "violations": list(self.violations),
        }


Measure = Callable[[int, int], Measurement]


def sweep_axes(
    *,
    model: str,
    sessions_axis: Iterable[int],
    prefill_axis: Iterable[int],
    measure: Measure,
    min_gain_ratio: float = DEFAULT_MIN_GAIN_RATIO,
) -> SweepResult:
    """Sweep both axes jointly and keep the best measured knee.

    The sessions axis stops at the knee of the current prefill budget: once the
    batch saturates, wider batches can only add memory pressure. The best
    (prefill, sessions) pair is the curve whose knee delivers the most aggregate
    total tokens per second.
    """
    sessions = sorted(set(sessions_axis))
    prefills = sorted(set(prefill_axis))
    if not sessions or not prefills:
        raise SweepError("both sweep axes need at least one value")
    violations: list[str] = []
    curves: list[AxisResult] = []
    for prefill in prefills:
        curve = AxisResult(prefill_tokens=prefill)
        for width in sessions:
            curve.points.append(measure(prefill, width))
            knee = detect_knee(curve.points, min_gain_ratio)
            if knee.saturated:
                break
        curve.knee = detect_knee(curve.points, min_gain_ratio)
        curves.append(curve)
    saturated = [curve for curve in curves if curve.knee and curve.knee.saturated]
    if not saturated:
        violations.append(
            "no curve saturated within the sessions axis: the knee is beyond "
            "the measured batch widths, extend --sessions-axis"
        )
        best_curve = max(
            curves, key=lambda curve: curve.knee.knee_total_tokens_per_second
        )
    else:
        best_curve = max(
            saturated, key=lambda curve: curve.knee.knee_total_tokens_per_second
        )
    return SweepResult(
        model=model,
        axes={"sessions": sessions, "prefillTokens": prefills},
        min_gain_ratio=min_gain_ratio,
        curves=curves,
        best=best_curve.knee,
        best_prefill_tokens=best_curve.prefill_tokens,
        violations=violations,
    )


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise SweepError(
                f"server exited during startup with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                base_url.rstrip("/") + "/health", timeout=2.0
            ):
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.5)


@contextlib.contextmanager
def serving_process(
    *,
    binary: Path,
    model_path: Path,
    sessions: int,
    prefill_tokens: int,
    extra_args: Sequence[str],
    startup_timeout: float,
):
    """Run one `gufo serve` point and yield its base URL."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gufo import serving_bench  # noqa: PLC0415 - optional import at use

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    # Server-wide options precede the `llm` subcommand; model and scheduling
    # options belong to it.
    command = [
        str(binary),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--sessions",
        str(sessions),
        "llm",
        "--model",
        str(model_path),
        "--prefill-chunk",
        str(prefill_tokens),
        *extra_args,
    ]
    process = subprocess.Popen(command)
    try:
        _wait_ready(base_url, process, startup_timeout)
        model = None
        try:
            model = serving_bench.discover_model(base_url, 30.0)
        except RuntimeError:
            # The smoke may be disabled; the sweep still benchmarks the model
            # ID the caller asked for through --model-id below.
            model = None
        yield base_url, model
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        # One point's weights and state pool live in unified memory. Wait for
        # the driver to release the previous process's device mappings before
        # loading the same model again.
        _wait_device_idle()
        time.sleep(SETTLE_SECONDS)


def benchmark_point(
    *,
    base_url: str,
    model: str,
    sessions: int,
    max_tokens: int,
    prompt_repeat: int,
    warmup_rounds: int,
    repetitions: int,
    timeout_seconds: float,
    fingerprint: dict[str, object] | None,
    notes: list[str] | None = None,
) -> Measurement:
    """Measure one point with the canonical serving benchmark."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gufo import serving_bench  # noqa: PLC0415 - optional import at use

    report = serving_bench.run_benchmark(
        base_url=base_url,
        model=model,
        prompt=serving_bench.DEFAULT_PROMPT,
        workload_id="canonical-serving-v1",
        max_tokens=max_tokens,
        temperature=0.0,
        # The measured batch is the admitted width: one synchronized wave.
        concurrency_levels=[sessions],
        warmup_rounds=warmup_rounds,
        repetitions=repetitions,
        timeout_seconds=timeout_seconds,
        fingerprint=fingerprint
        or {"fingerprintId": "0" * 64, "canonical": {"schemaVersion": "1.0.0"}},
        source_revision="unknown",
        source_dirty=True,
        prompt_repeat=prompt_repeat,
        notes=list(notes or []),
    )
    for item in report.get("warnings", []):
        if "exceeded" in item:
            raise SweepError(f"concurrency exceeded the batch: {item}")
    return _measurement_from_report(report)


def _measurement_from_report(report: dict[str, object]) -> Measurement:
    results = report.get("results")
    if not isinstance(results, dict) or not results:
        raise SweepError("benchmark report has no measured rounds")
    round_key = next(iter(results))
    aggregate = results[round_key].get("aggregate", {})
    total = aggregate.get("total_tokens_per_second", {}).get("overall")
    output = aggregate.get("output_tokens_per_second", {}).get("overall")
    decode = aggregate.get("decode_tokens_per_second", {}).get("overall")
    prefill = aggregate.get("prefill_tokens_per_second", {}).get("overall")
    if total is None or output is None:
        raise SweepError("benchmark report lacks aggregate throughput")
    workload = report.get("workload", {})
    concurrency = workload.get("concurrency") or [0]
    sessions = int(concurrency[0])
    if decode is None:
        # The PRD's primary objective is the summed decode rate of all requests
        # admitted in one batch. serving_bench reports it per measured cohort
        # instead of in the aggregate block, so reuse the canonical table rate.
        from gufo.model_bench.render import _serving_rate  # noqa: PLC0415

        decode = _serving_rate(report, sessions)
    return Measurement(
        sessions=sessions,
        prefill_tokens=0,
        total_tokens_per_second=float(total),
        output_tokens_per_second=float(output),
        decode_tokens_per_second=None if decode is None else float(decode),
        prefill_tokens_per_second=None if prefill is None else float(prefill),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gufo-serving-sweep",
        description=(
            "Sweep max_num_seqs (--sessions) and max_num_batched_tokens "
            "(--prefill-chunk) together and report the knee of the aggregate "
            "throughput curve."
        ),
    )
    parser.add_argument("--gufo", type=Path, default=Path("./result/bin/gufo"))
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Model checkpoint directory passed to `gufo serve --model`",
    )
    parser.add_argument("--model-id", required=True, help="Served model ID")
    parser.add_argument(
        "--sessions-axis",
        type=parse_axis,
        default=parse_axis("1,2,4,8,16,32,64,128"),
        help="Batch widths to measure (default: 1,2,4,8,16,32,64,128)",
    )
    parser.add_argument(
        "--prefill-axis",
        type=parse_axis,
        default=parse_axis("512,2048,8192"),
        help="Batched-token budgets to measure (default: 512,2048,8192)",
    )
    parser.add_argument(
        "--min-gain",
        type=float,
        default=DEFAULT_MIN_GAIN_RATIO,
        help="Fractional gain that keeps a batch step on the rising part",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-repeat", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional `gufo serve` argument (repeatable)",
    )
    parser.add_argument("--fingerprint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT / "tools"))
    from gufo import serving_bench  # noqa: PLC0415 - optional import at use

    fingerprint = None
    if args.fingerprint is not None:
        fingerprint = json.loads(args.fingerprint.read_text(encoding="utf-8"))
    extra = list(args.extra_arg)
    cached: dict[tuple[int, int], Measurement] = {}

    def measure(prefill: int, sessions: int) -> Measurement:
        with serving_process(
            binary=args.gufo,
            model_path=args.model_path,
            sessions=sessions,
            prefill_tokens=prefill,
            extra_args=extra,
            startup_timeout=args.startup_timeout,
        ) as (base_url, served):
            point = benchmark_point(
                base_url=base_url,
                model=served or args.model_id,
                sessions=sessions,
                max_tokens=args.max_tokens,
                prompt_repeat=args.prompt_repeat,
                warmup_rounds=args.warmup,
                repetitions=args.repetitions,
                timeout_seconds=args.timeout,
                fingerprint=fingerprint,
                notes=[
                    f"--sessions {sessions}",
                    f"--prefill-chunk {prefill}",
                ],
            )
        measured = Measurement(
            sessions=sessions,
            prefill_tokens=prefill,
            total_tokens_per_second=point.total_tokens_per_second,
            output_tokens_per_second=point.output_tokens_per_second,
            decode_tokens_per_second=point.decode_tokens_per_second,
            prefill_tokens_per_second=point.prefill_tokens_per_second,
        )
        cached[(prefill, sessions)] = measured
        return measured

    try:
        result = sweep_axes(
            model=args.model_id,
            sessions_axis=args.sessions_axis,
            prefill_axis=args.prefill_axis,
            measure=measure,
            min_gain_ratio=args.min_gain,
        )
    except SweepError as error:
        print(f"sweep failed: {error}", file=sys.stderr)
        return 1
    report = result.as_dict()
    report["measuredAt"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["runId"] = uuid.uuid4().hex
    if args.output is not None:
        from gufo.serving_bench import atomic_json  # noqa: PLC0415

        atomic_json(args.output, report)
    if args.json or args.output is None:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        best = result.best
        if best is not None:
            print(
                f"{result.model}: knee at {best.knee_sessions} sessions "
                f"x {result.best_prefill_tokens} prefill tokens, "
                f"{best.knee_total_tokens_per_second:.2f} total tok/s"
            )
    for violation in result.violations:
        print(f"violation: {violation}", file=sys.stderr)
    return 0 if not result.violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
