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


def to_json_safe(value: Any) -> Any:
    """Return a value containing only types representable in strict JSON.

    RFC 8259 has no literal for ``Infinity`` / ``NaN``, so non-finite floats
    become ``None`` instead of leaking Python's non-standard extensions into
    logs, stdout summaries and report files. Numpy scalars/arrays, Paths and
    Enums are flattened to plain Python values.
    """
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items() if k != "_latencies"}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return to_json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value") and hasattr(value, "name"):  # Enum
        return to_json_safe(value.value)
    return value


def dumps_json(value: Any, **kwargs: Any) -> str:
    """``json.dumps`` that always emits RFC 8259 compliant JSON.

    ``allow_nan=False`` is a safety net: even if a non-finite value slips past
    ``to_json_safe`` (e.g. a custom object's ``default`` hook), it raises
    instead of silently writing invalid JSON.
    """
    kwargs.setdefault("default", _json_default)
    return json.dumps(
        to_json_safe(value), ensure_ascii=False, allow_nan=False, **kwargs
    )


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
        self._fh.write(dumps_json(record))
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
    # Kept as a private alias: frame records go through the same strict-JSON
    # conversion as every other serialised output.
    return to_json_safe(value)


def log_event(code: str, message: str = "", **fields: Any) -> None:
    logger.warning("%s %s", code, message, extra={"code": code, **fields})


def utc_ms() -> float:
    return time.time() * 1000.0


__all__ = [
    "logger",
    "setup_logging",
    "JsonlFrameLogger",
    "to_json_safe",
    "dumps_json",
    "log_event",
    "utc_ms",
]
