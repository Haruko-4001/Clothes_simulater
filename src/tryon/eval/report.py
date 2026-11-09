"""Evaluation report: real metrics only, TODO placeholders for unmeasured data.

IMPL: src/tryon/eval/report.py
No fabricated "already measured" numbers are permitted (SPEC §15 / §17.10).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, Config
from ..logging import dumps_json, logger


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


def build_report(
    cfg: Config,
    *,
    log_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    log_path = Path(log_path) if log_path is not None else REPO_ROOT / "var" / "logs" / "run.jsonl"
    output_dir = Path(output_dir) if output_dir is not None else REPO_ROOT / "var" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(log_path)
    verdicts = [r.get("framing", {}).get("verdict", "") for r in records]
    material_states = [r.get("material_state", "") for r in records]
    penetrations = [int(r.get("physics", {}).get("penetration_count", 0)) for r in records]
    attached = [int(r.get("physics", {}).get("attached_count", 0)) for r in records]
    final_timing = dict(records[-1].get("timing", {})) if records else {}
    # Internal-only meter: never expose it through an external metric report.
    final_timing.pop("generation_throughput_fps", None)

    report: dict[str, Any] = {
        "spec_version": "v2.0",
        "mode": cfg.get("mode"),
        "log_path": str(log_path),
        "n_frames_logged": len(records),
        "mock_run_counters": {
            "framing_compliance_rate": (
                sum(1 for v in verdicts if v == "OK") / len(verdicts) if verdicts else None
            ),
            "verdict_histogram": {v: verdicts.count(v) for v in sorted(set(verdicts))},
            "material_state_histogram": {
                s: material_states.count(s) for s in sorted(set(material_states))
            },
            "max_penetration_count": max(penetrations) if penetrations else None,
            "max_attached_count": max(attached) if attached else None,
            "timing_final": final_timing,
            "note": "generation_throughput_fps is logged internally only and is redacted here",
        },
        "target_hardware_measurements": {
            "status": "TODO — no numbers are claimed until measured on target hardware",
            "rows": [{"metric": m, "value": v, "note": n} for m, v, n in TARGET_HARDWARE_ROWS],
        },
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
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("eval report written to %s and %s", json_path, md_path)
    return report


__all__ = ["build_report", "TARGET_HARDWARE_ROWS"]
