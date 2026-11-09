"""Regression: serialised outputs must be RFC 8259 compliant JSON.

A previous version printed ``Infinity`` in the main-loop summary (from a
singular Fisher matrix). Python's json accepts that extension, but JS
JSON.parse / jq / serde reject it outright.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
import subprocess
import sys

import numpy as np

from tryon.logging import JsonlFrameLogger, dumps_json, to_json_safe

ROOT = Path(__file__).resolve().parents[1]


def _strict_loads(text: str):
    def reject(constant: str):
        raise ValueError(f"non-standard JSON constant: {constant}")

    return json.loads(text, parse_constant=reject)


class _Colour(Enum):
    RED = "red"


def test_non_finite_floats_become_null_and_output_is_strict_json():
    payload = {
        "calibration_last_cond": float("inf"),
        "neg": float("-inf"),
        "nan": float("nan"),
        "nested": {"values": [1.5, float("inf"), None]},
        "untouched": 3,
    }
    text = dumps_json(payload, indent=2)
    assert "Infinity" not in text and "NaN" not in text
    parsed = _strict_loads(text)
    assert parsed["calibration_last_cond"] is None
    assert parsed["neg"] is None
    assert parsed["nan"] is None
    assert parsed["nested"]["values"] == [1.5, None, None]
    assert parsed["untouched"] == 3


def test_to_json_safe_flattens_numpy_path_and_enum():
    cleaned = to_json_safe(
        {
            "arr": np.array([1.0, float("inf")]),
            "scalar": np.float64(2.5),
            "path": Path("var/logs/run.jsonl"),
            "enum": _Colour.RED,
        }
    )
    assert cleaned == {
        "arr": [1.0, None],
        "scalar": 2.5,
        "path": str(Path("var/logs/run.jsonl")),
        "enum": "red",
    }
    _strict_loads(dumps_json(cleaned))


def test_frame_logger_never_writes_non_standard_constants(tmp_path: Path):
    log = tmp_path / "frames.jsonl"
    with JsonlFrameLogger(log) as frame_logger:
        record = frame_logger.make_record(
            frame_id=1,
            pts_ms=0.0,
            asset_id="g_test",
            asset_version=0,
            timing={"camera_capture_fps": 30.0},
            framing_verdict="OK",
            visible_keypoint_ratio=1.0,
            guide_text="",
            material_state="AI_PRIOR",
            back_geometry_confidence=0.35,
            attached_count=0,
            penetration_count=0,
            energy=float("inf"),
        )
        frame_logger.write(record)
        # write() must be safe even for callers that bypass make_record().
        frame_logger.write({"pts_ms": 1.0, "energy": float("nan")})

    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert _strict_loads(lines[0])["physics"]["energy"] is None
    assert _strict_loads(lines[1])["energy"] is None


def test_mock_run_stdout_summary_is_strict_json(tmp_path: Path):
    """The exact defect: main-loop stdout used to contain bare ``Infinity``."""
    log = tmp_path / "run.jsonl"
    proc = subprocess.run(
        [
            sys.executable, "-m", "app.main", "--mode", "mock",
            "--frames", "8", "--headless", "--log", str(log),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Infinity" not in proc.stdout and "NaN" not in proc.stdout
    parsed = _strict_loads(proc.stdout)
    assert parsed["frames"] == 8
    assert "calibration_last_cond" in parsed
