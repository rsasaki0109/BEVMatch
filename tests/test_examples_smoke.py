"""v1.0 reproducible demo suite: representative examples run without error."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("mod_name", [
    "examples.run_demo",
    "examples.run_ros_replay",
    "examples.run_multimodal",
])
def test_example_runs(mod_name):
    mod = importlib.import_module(mod_name)
    mod.main()  # must not raise


def test_demo_suite_runner_fast():
    runner = importlib.import_module("examples.run_all_demos")
    # --fast skips the heavy benchmark; expect zero failures
    assert runner.main(["--fast"]) == 0
