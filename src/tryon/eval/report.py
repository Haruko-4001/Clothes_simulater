"""Evaluation report: real metrics only, TODO placeholders for unmeasured data.

IMPL: src/tryon/eval/report.py
No fabricated "already measured" numbers are permitted (SPEC §15 / §17.10).

Every metric defined in SPEC §7 is implemented in ``metrics.py`` and wired
here: framework counters are computed from the JSONL log, while ground-truth
metrics are exposed through :func:`evaluate_ground_truth`. Target-hardware
values stay ``null / NOT_MEASURED`` until they are actually measured.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, Config
from ..logging import dumps_json, logger
from . import metrics

# (metric, markdown placeholder, note). Kept as display triples for report.md;
# report.json writes value=null + status=NOT_MEASURED instead of a string.
TARGET_HARDWARE_ROWS = [
    ("capture_to_display p50/p95 (ms)", "TODO", "目标硬件实测"),
    ("geometry IoU / CD / normal error", "TODO", "需真实资产与标注"),
    ("ADE@0.4s / FDE@0.4s", "TODO", "需真实交互轨迹"),
    ("penetration count", "TODO", "需目标硬件 + 真实人体"),
    ("parameter error ε_Θ", "TODO", "合成数据可测，待运行"),
    ("peak VRAM (GB)", "TODO", "需目标 GPU"),
    ("framing compliance rate", "TODO", "需真实摄像头数据"),
    ("degrade count", "TODO", "需真实摄像头数据"),
]

# §7 / §15 Phase 5: every metric family has a named implementation.
METRIC_FAMILIES = (
    ("iou", "SPEC §7 IoU"),
    ("chamfer_distance", "SPEC §7 Chamfer CD"),
    ("normal_error", "SPEC §7 法向误差"),
    ("ade", "SPEC §7 ADE@H"),
    ("fde", "SPEC §7 FDE@H"),
    ("penetration_rate", "SPEC §7 穿模率"),
    ("parameter_error", "SPEC §7 ε_Θ"),
    ("energy_decay_ok", "SPEC §7 能量衰减"),
    ("convergence", "SPEC §7 收敛 / G3"),
    ("latency_percentiles", "SPEC §7 p50/p95 延迟"),
    ("framing_compliance_rate", "SPEC §7 取景合规率"),
)


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def evaluate_ground_truth(
    *,
    mask_pred: Any = None,
    mask_true: Any = None,
    points_pred: Any = None,
    points_true: Any = None,
    normals_pred: Any = None,
    normals_true: Any = None,
    trajectory_pred: Any = None,
    trajectory_true: Any = None,
    horizon: int = 12,
    theta_hat: Any = None,
    theta_star: Any = None,
    penetration_counts: Any = None,
    n_vertices: int | None = None,
    energies: Any = None,
    gamma: float | None = None,
    dt: float | None = None,
    variances: Any = None,
    latency_samples_ms: Any = None,
) -> dict[str, Any]:
    """Compute every §7 metric for which ground truth was actually supplied.

    Absent inputs yield ``null``. This is a pure function: it never writes
    numbers into the target-hardware table and never fabricates a measurement.
    Callers may use synthetic ground truth, in which case the result belongs
    under ``synthetic_validation`` — never under ``measured_numbers``.
    """
    out: dict[str, Any] = {
        "iou": None,
        "chamfer_distance": None,
        "normal_error": None,
        "ade": None,
        "fde": None,
        "penetration_rate": None,
        "parameter_error": None,
        "energy_decay_ok": None,
        "convergence": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
    }
    if mask_pred is not None and mask_true is not None:
        out["iou"] = _finite(metrics.iou(mask_pred, mask_true))
    if points_pred is not None and points_true is not None:
        out["chamfer_distance"] = _finite(metrics.chamfer_distance(points_pred, points_true))
    if normals_pred is not None and normals_true is not None:
        out["normal_error"] = _finite(metrics.normal_error(normals_pred, normals_true))
    if trajectory_pred is not None and trajectory_true is not None:
        out["ade"] = _finite(metrics.ade(trajectory_pred, trajectory_true, horizon))
        out["fde"] = _finite(metrics.fde(trajectory_pred, trajectory_true, horizon))
    if penetration_counts is not None and n_vertices:
        out["penetration_rate"] = _finite(
            metrics.penetration_rate(penetration_counts, int(n_vertices))
        )
    if theta_hat is not None and theta_star is not None:
        out["parameter_error"] = _finite(metrics.parameter_error(theta_hat, theta_star))
    if energies is not None and gamma is not None and dt is not None:
        out["energy_decay_ok"] = metrics.energy_decay_ok(list(energies), gamma, dt)
    if variances is not None:
        out["convergence"] = metrics.convergence(list(variances))
    if latency_samples_ms is not None:
        p50, p95 = metrics.latency_percentiles(list(latency_samples_ms))
        out["latency_p50_ms"] = _finite(p50)
        out["latency_p95_ms"] = _finite(p95)
    out["note"] = (
        "computed only from caller-supplied ground truth; "
        "NOT a target-hardware measurement; never written to measured_numbers"
    )
    return out


def build_report(
    cfg: Config,
    *,
    log_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ground_truth: dict[str, Any] | None = None,
    n_vertices: int | None = None,
    latency_samples_ms: list[float] | None = None,
) -> dict[str, Any]:
    log_path = Path(log_path) if log_path is not None else REPO_ROOT / "var" / "logs" / "run.jsonl"
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "var" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(log_path)
    verdicts = [r.get("framing", {}).get("verdict", "") for r in records]
    material_states = [r.get("material_state", "") for r in records]
    penetrations = [int(r.get("physics", {}).get("penetration_count", 0)) for r in records]
    attached = [int(r.get("physics", {}).get("attached_count", 0)) for r in records]
    energies = [float(r.get("physics", {}).get("energy", 0.0)) for r in records]
    degrade_events = sum(
        1
        for r in records
        for e in r.get("events", [])
        if "DEGRADE" in str(e.get("code", "")).upper()
    )
    final_timing = dict(records[-1].get("timing", {})) if records else {}
    # Internal-only meter: never expose it through an external metric report.
    final_timing.pop("generation_throughput_fps", None)

    gamma = float(cfg.require("physics.energy_decay_gamma"))
    dt = float(cfg.require("math.dt"))
    counters: dict[str, Any] = {
        "framing_compliance_rate": (
            _finite(metrics.framing_compliance_rate(verdicts)) if verdicts else None
        ),
        "verdict_histogram": {v: verdicts.count(v) for v in sorted(set(verdicts))},
        "material_state_histogram": {
            s: material_states.count(s) for s in sorted(set(material_states))
        },
        "max_penetration_count": max(penetrations) if penetrations else None,
        "max_attached_count": max(attached) if attached else None,
        "degrade_event_count": degrade_events,
        "energy_decay_ok": metrics.energy_decay_ok(energies, gamma, dt) if energies else None,
        "timing_final": final_timing,
        "note": (
            "mock framework counters, NOT target-hardware measurements; "
            "generation_throughput_fps is logged internally only and redacted here"
        ),
    }
    if n_vertices:
        counters["penetration_rate"] = _finite(
            metrics.penetration_rate(penetrations, int(n_vertices))
        )
    if latency_samples_ms:
        p50, p95 = metrics.latency_percentiles(latency_samples_ms)
        counters["latency_p50_ms"] = _finite(p50)
        counters["latency_p95_ms"] = _finite(p95)

    report: dict[str, Any] = {
        "spec_version": "v2.0",
        "mode": cfg.get("mode"),
        "log_path": str(log_path),
        "n_frames_logged": len(records),
        "mock_run_counters": counters,
        "target_hardware_measurements": {
            "status": "NOT_MEASURED — no numbers are claimed until measured on target hardware",
            "rows": [
                {"metric": m, "value": None, "status": "NOT_MEASURED", "note": n}
                for m, _placeholder, n in TARGET_HARDWARE_ROWS
            ],
        },
        "metric_families": {
            "implemented_in": "src/tryon/eval/metrics.py",
            "formulas": "SPEC §7 (definitions precede implementation)",
            "symbols": [
                {"symbol": s, "definition": d, "status": "AVAILABLE"}
                for s, d in METRIC_FAMILIES
            ],
        },
        "synthetic_validation": (
            evaluate_ground_truth(**ground_truth) if ground_truth is not None else None
        ),
        "measured_numbers": [],
    }

    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"
    json_path.write_text(dumps_json(report, indent=2), encoding="utf-8")

    lines = [
        "# Evaluation Report (SPEC v2.0)",
        "",
        f"- mode: `{report['mode']}`",
        f"- frames logged: {report['n_frames_logged']}",
        f"- log: `{log_path}`",
        "",
        "## Mock framework counters (NOT target-hardware measurements)",
        "```json",
        dumps_json(report["mock_run_counters"], indent=2),
        "```",
        "",
        "## Metric implementations (SPEC §7)",
        "",
        "All metrics are implemented in `src/tryon/eval/metrics.py` and wired via",
        "`evaluate_ground_truth()`; absent ground truth yields `null`.",
        "",
        "## Target-hardware table (empty by design — fill only with real measurements)",
        "",
        "| Metric | Value | Note |",
        "|---|---|---|",
    ]
    for m, v, n in TARGET_HARDWARE_ROWS:
        lines.append(f"| {m} | {v} | {n} |")
    lines += [
        "",
        "> `generation_throughput_fps` is recorded only; it is never an external metric.",
        "> p50/p95 are empirical percentiles of `Frame.pts_ms → display`, never means.",
        "> `measured_numbers` stays empty until target-hardware results exist.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("eval report written to %s and %s", json_path, md_path)
    return report


__all__ = [
    "build_report",
    "evaluate_ground_truth",
    "TARGET_HARDWARE_ROWS",
    "METRIC_FAMILIES",
]
