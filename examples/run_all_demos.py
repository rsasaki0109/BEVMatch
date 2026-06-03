"""Reproducible demo suite (§21 v1.0).

    python examples/run_all_demos.py [--fast]

Runs every BEVMatch example end-to-end and reports pass/fail. With --fast, skips
the heavier benchmark suite. Used as the release gate's reproducibility check.
"""

from __future__ import annotations

import importlib
import sys
import time

# (module, needs_ros2, heavy)
DEMOS = [
    ("examples.run_demo", False, False),
    ("examples.run_retrieval_eval", False, False),
    ("examples.run_alignment_eval", False, False),
    ("examples.run_change_eval", False, False),
    ("examples.run_map_validation", False, False),
    ("examples.run_ros_replay", False, False),
    ("examples.run_autoware_nav2", False, False),
    ("examples.run_multimodal", False, False),
    ("examples.run_benchmark_suite", False, True),
]


def _has_rclpy() -> bool:
    try:
        importlib.import_module("rclpy")
        return True
    except Exception:
        return False


def main(argv: list[str]) -> int:
    fast = "--fast" in argv
    results = []
    for mod_name, needs_ros2, heavy in DEMOS:
        if heavy and fast:
            results.append((mod_name, "skipped (fast)"))
            continue
        if needs_ros2 and not _has_rclpy():
            results.append((mod_name, "skipped (no ROS2)"))
            continue
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
            results.append((mod_name, f"OK ({time.time() - t0:.1f}s)"))
        except Exception as exc:  # noqa: BLE001 - report and continue
            results.append((mod_name, f"FAIL: {type(exc).__name__}: {exc}"))

    print("\n" + "=" * 60)
    print("Demo suite summary")
    print("=" * 60)
    failed = 0
    for name, status in results:
        if status.startswith("FAIL"):
            failed += 1
        print(f"  {name:<32} {status}")
    print("=" * 60)
    print(f"{len(results) - failed}/{len(results)} ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
