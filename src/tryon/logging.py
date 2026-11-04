"""Structured JSONL logging (SPEC Appendix D).

IMPL: src/tryon/logging.py
One JSON object per line. ``generation_throughput_fps`` must never be exposed
to external metric endpoints.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


logger = logging.getLogger("tryon")


def setup_logging(level: int = logging.INFO) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    return str(obj)


class JsonlFrameLogger:
    """Writes one schema-complete record per displayed frame."""

    REQUIRED_TOP_LEVEL = (
        "frame_id",
        "pts_ms",
        "asset_id",
        "asset_version",
        "timing",
        "framing",
        "material_state",
        "confidence",
        "physics",
        "events",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self._closed = False

    @staticmethod
    def make_record(
        *,
        frame_id: int,
        pts_ms: float,
        asset_id: str,
        asset_version: int,
        timing: Mapping[str, float],
        framing_verdict: str,
        visible_keypoint_ratio: float,
        guide_text: str,
        material_state: str,
        back_geometry_confidence: float,
        attached_count: int,
        penetration_count: int,
        energy: float,
        events: list[dict[str, Any]] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        timing_dict = dict(timing)
        # Appendix D says 9 timing fields; §9.3 TimingBook defines the 8
        # authoritative fields. ``max`` is derived for logging only.
        samples = timing_dict.pop("_latencies", None)
        if samples is not None:
            arr = np.asarray(samples, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            timing_dict["capture_to_display_ms_max"] = (
                float(arr.max()) if arr.size else 0.0
            )
        record: dict[str, Any] = {
            "frame_id": int(frame_id),
            "pts_ms": float(pts_ms),
            "asset_id": str(asset_id),
            "asset_version": int(asset_version),
            "timing": timing_dict,
            "framing": {
                "verdict": str(framing_verdict),
                "visible_keypoint_ratio": float(visible_keypoint_ratio),
                "guide_text": str(guide_text),
            },
            "material_state": str(material_state),
            "confidence": {"back_geometry": float(back_geometry_confidence)},
            "physics": {
                "attached_count": int(attached_count),
                "penetration_count": int(penetration_count),
                "energy": float(energy),
            },
            "events": list(events or []),
        }
        if extra:
            record.update(dict(extra))
        missing = [k for k in JsonlFrameLogger.REQUIRED_TOP_LEVEL if k not in record]
        if missing:
            raise ValueError(f"frame log record missing keys: {missing}")
        return _sanitise(record)

    def write(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("logger is closed")
        if math.isnan(float(record.get("pts_ms", 0.0))):
            raise ValueError("pts_ms is NaN")
        self._fh.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        self._fh.write("\n")

    def close(self) -> None:
        if not self._closed:
            self._fh.flush()
            self._fh.close()
            self._closed = True

    def __enter__(self) -> "JsonlFrameLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _sanitise(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _sanitise(v) for k, v in value.items() if k != "_latencies"}
    if isinstance(value, (list, tuple)):
        return [_sanitise(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def log_event(code: str, message: str = "", **fields: Any) -> None:
    logger.warning("%s %s", code, message, extra={"code": code, **fields})


def utc_ms() -> float:
    return time.time() * 1000.0


__all__ = ["logger", "setup_logging", "JsonlFrameLogger", "log_event", "utc_ms"]
