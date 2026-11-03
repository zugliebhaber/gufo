#!/usr/bin/env python3
"""Hosted contracts for the joint batch/prefill sweep and its knee detection."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from gufo import serving_sweep


def point(sessions, total, prefill=512):
    return serving_sweep.Measurement(
        sessions=sessions,
        prefill_tokens=prefill,
        total_tokens_per_second=total,
        output_tokens_per_second=total / 2,
    )


class AxisTest(unittest.TestCase):
    def test_list_and_range_forms(self):
        self.assertEqual(serving_sweep.parse_axis("8, 1,4"), [1, 4, 8])
        self.assertEqual(serving_sweep.parse_axis("1:8:2"), [1, 3, 5, 7])
        self.assertEqual(serving_sweep.parse_axis("16:32"), list(range(16, 33)))

    def test_rejects_empty_zero_and_bad_ranges(self):
        for value in ("", "0", "a", "8:1", "1:2:0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                serving_sweep.parse_axis(value)


class KneeTest(unittest.TestCase):
    def test_knee_is_the_last_step_that_pays(self):
        knee = serving_sweep.detect_knee(
            [
                point(1, 20.0),
                point(2, 40.0),
                point(4, 80.0),
                point(8, 82.0),
                point(16, 80.0),
            ]
        )
        self.assertEqual(knee.knee_sessions, 4)
        self.assertEqual(knee.peak_sessions, 8)
        self.assertTrue(knee.saturated)
        self.assertTrue(knee.monotone)
        self.assertIn("4->8", knee.reason)

    def test_decline_is_saturation_at_the_peak(self):
        knee = serving_sweep.detect_knee(
            [point(1, 20.0), point(2, 30.0), point(4, 29.0), point(8, 60.0)]
        )
        # A wider batch with less aggregate throughput is past the knee; the
        # early stop means the later 60 is never measured (nor smoothed in).
        self.assertTrue(knee.saturated)
        self.assertTrue(knee.monotone)
        self.assertEqual(knee.knee_sessions, 2)
        self.assertIn("2->4", knee.reason)

    def test_still_rising_axis_is_unsaturated(self):
        knee = serving_sweep.detect_knee([point(1, 10.0), point(2, 20.0)])
        self.assertFalse(knee.saturated)
        self.assertTrue(knee.monotone)
        self.assertEqual(knee.knee_sessions, 2)

    def test_threshold_decides_the_knee(self):
        points = [point(1, 100.0), point(2, 105.0), point(4, 125.0)]
        strict = serving_sweep.detect_knee(points, min_gain_ratio=0.1)
        loose = serving_sweep.detect_knee(points, min_gain_ratio=0.01)
        self.assertEqual(strict.knee_sessions, 1)
        self.assertEqual(loose.knee_sessions, 4)
        self.assertTrue(loose.saturated is False)


class SweepTest(unittest.TestCase):
    def test_inner_axis_stops_at_the_knee(self):
        calls = []

        def measure(prefill, sessions):
            calls.append((prefill, sessions))
            # 512-token chunks saturate at 4, 2048 at 8 (and pay more).
            if prefill == 512:
                total = 100.0 * min(sessions, 4) + (0.5 if sessions > 4 else 0.0)
            else:
                total = 130.0 * min(sessions, 8) + (0.5 if sessions > 8 else 0.0)
            return point(sessions, total, prefill)

        result = serving_sweep.sweep_axes(
            model="qwen3.8-27b",
            sessions_axis=[1, 2, 4, 8, 16],
            prefill_axis=[512, 2048],
            measure=measure,
        )
        # 512: 1,2,4,8 (8 is the saturating step) ; 2048: 1,2,4,8,16
        self.assertEqual(
            [width for prefill, width in calls if prefill == 512], [1, 2, 4, 8]
        )
        self.assertEqual(
            [width for prefill, width in calls if prefill == 2048],
            [1, 2, 4, 8, 16],
        )
        self.assertEqual(result.best_prefill_tokens, 2048)
        self.assertEqual(result.best.knee_sessions, 8)
        self.assertEqual(result.violations, [])
        self.assertTrue(result.best.saturated)

    def test_each_target_model_gets_a_knee(self):
        curves = {
            "qwen3.8-27b": (19.0, 100.0),
            "qwen3.8-flash-next": (45.0, 150.0),
            "deepseek-v4-flash": (24.0, 80.0),
        }
        knees = {}
        for model, (_, target) in curves.items():
            # Throughput grows with the batch until the target, then flattens.
            def measure(prefill, sessions, target=target):
                # Rising until the target, then flat: the batch saturates.
                return point(sessions, min(sessions * target / 64.0, target), prefill)

            result = serving_sweep.sweep_axes(
                model=model,
                sessions_axis=[1, 2, 4, 8, 16, 32, 64, 128],
                prefill_axis=[512],
                measure=measure,
            )
            self.assertEqual(result.violations, [], model)
            self.assertIsNotNone(result.best, model)
            knees[model] = result.best.knee_sessions
        self.assertEqual(set(knees), set(curves))
        for model, sessions in knees.items():
            self.assertGreater(sessions, 1, model)
            self.assertLessEqual(sessions, 128, model)

    def test_unsaturated_axis_is_a_violation(self):
        result = serving_sweep.sweep_axes(
            model="m",
            sessions_axis=[1, 2],
            prefill_axis=[512],
            measure=lambda prefill, sessions: point(
                sessions, 10.0 * sessions, prefill
            ),
        )
        self.assertTrue(any("saturated within" in v for v in result.violations))

    def test_report_round_trips_as_json(self):
        result = serving_sweep.sweep_axes(
            model="m",
            sessions_axis=[1, 2, 4],
            prefill_axis=[512],
            measure=lambda prefill, sessions: point(
                sessions, [10.0, 30.0, 30.1][[1, 2, 4].index(sessions)], prefill
            ),
        )
        report = result.as_dict()
        encoded = json.dumps(report)
        self.assertIn("gufo.serving-sweep.v1", encoded)
        self.assertEqual(report["result"]["kneeSessions"], 2)
        self.assertEqual(report["axes"]["sessions"], [1, 2, 4])

    def test_benchmark_report_maps_to_a_measurement(self):
        report = {
            "workload": {"concurrency": [8]},
            "results": {
                "c8": {
                    "aggregate": {
                        "total_tokens_per_second": {"overall": 423.9},
                        "output_tokens_per_second": {"overall": 200.5},
                        "decode_tokens_per_second": {"overall": 210.0},
                        "prefill_tokens_per_second": {"overall": 1010.0},
                    }
                }
            },
        }
        measurement = serving_sweep._measurement_from_report(report)
        self.assertEqual(measurement.sessions, 8)
        self.assertAlmostEqual(measurement.total_tokens_per_second, 423.9)
        self.assertAlmostEqual(measurement.prefill_tokens_per_second, 1010.0)

    def test_aggregate_decode_is_summed_per_batch(self):
        report = {
            "workload": {"concurrency": [4]},
            "results": {
                "c4": {
                    "aggregate": {
                        "total_tokens_per_second": {"overall": 300.0},
                        "output_tokens_per_second": {"overall": 120.0},
                    },
                    "rounds": [{"sampleCount": 4}, {"sampleCount": 4}],
                    "samples": [
                        {"decode_tokens_per_second": value}
                        for value in (40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0)
                    ],
                }
            },
        }
        measurement = serving_sweep._measurement_from_report(report)
        self.assertAlmostEqual(measurement.decode_tokens_per_second, 160.0)
        self.assertIsNone(
            serving_sweep._measurement_from_report(
                {"workload": {"concurrency": [4]},
                 "results": {"c4": {
                     "aggregate": {
                         "total_tokens_per_second": {"overall": 300.0},
                         "output_tokens_per_second": {"overall": 120.0},
                     }}}}
            ).decode_tokens_per_second,
            "a report without decode timings must not invent the metric",
        )

    def test_missing_aggregate_is_rejected(self):
        with self.assertRaises(serving_sweep.SweepError):
            serving_sweep._measurement_from_report({"results": {}})


class ArtifactTest(unittest.TestCase):
    def test_output_is_written_atomically(self):
        from gufo.serving_bench import atomic_json

        report = {"sweepSchema": "gufo.serving-sweep.v1"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sweep.json"
            atomic_json(path, report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)
            self.assertEqual(sorted(p.name for p in Path(directory).iterdir()),
                             ["sweep.json"])


if __name__ == "__main__":
    unittest.main()
