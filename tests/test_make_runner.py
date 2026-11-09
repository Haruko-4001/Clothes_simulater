"""Regression: ``make.py <target> KEY=VALUE`` must reach the child command.

Commands used to be built at import time, so the README-documented
``python scripts/make.py bootstrap-asset IMG=<path>`` silently used the
default garment instead of the supplied one.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAKE_PY = ROOT / "scripts" / "make.py"


@pytest.fixture(scope="module")
def make_runner():
    spec = importlib.util.spec_from_file_location("tryon_make_runner", MAKE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_overrides_separates_env_from_forwarded_args(make_runner):
    env, extra = make_runner.parse_overrides(
        ["IMG=custom.png", "--frames", "10", "LOG=run.jsonl"]
    )
    assert env == {"IMG": "custom.png", "LOG": "run.jsonl"}
    assert extra == ["--frames", "10"]


def test_bootstrap_command_uses_env_img(make_runner):
    cmd = make_runner.command_for("bootstrap-asset", {"IMG": "custom.png"})
    assert cmd[cmd.index("--image") + 1] == "custom.png"


def test_bootstrap_command_uses_default_without_img(make_runner):
    cmd = make_runner.command_for("bootstrap-asset", {})
    assert cmd[cmd.index("--image") + 1] == make_runner.DEFAULT_GARMENT


def test_documented_img_override_reaches_the_cli(tmp_path: Path):
    """End-to-end check of the exact README command.

    ``load_garment_image`` generates a sample when the path is missing, so a
    file appearing at the overridden path proves the override was forwarded.
    """
    image = tmp_path / "custom_garment.png"
    proc = subprocess.run(
        [sys.executable, str(MAKE_PY), "bootstrap-asset", f"IMG={image}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert image.exists(), "IMG override was ignored; the default garment was used"
