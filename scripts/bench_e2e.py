"""End-to-end A/B benchmark: numpy+PIL vs Warp+OpenCV on the real 300-frame loop.

The `local_benchmarks.end_to_end_*` numbers in STRUCTURED_REPORT.json were
produced by hand. This script reproduces the method with the actual app entry
point and writes a JSON snapshot, so the table is no longer a one-off manual
command.

What it measures (per profile, one full `app.main` run):
  * display_fps       — runner timing book (rolling counter)
  * p50_ms / p95_ms   — capture→display latency percentiles
  * physics_hz        — solver step rate
  * wall_s            — process wall time, including startup

It is **local-host engineering evidence**, not the SPEC §17.10 target-hardware
table. Host load makes ±30% run-to-run spread normal; use --repeats 3 and
compare medians.

Usage:
    python scripts/bench_e2e.py --frames 300 --repeats 3
    python scripts/bench_e2e.py --frames 60 --width 960 --height 540
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

PROFILES: dict[str, list[str]] = {
    "baseline_numpy_pil": [],
    "engine_warp_cv2": ["--physics-solver", "warp", "--render-backend", "opencv"],
}

_TIMING_KEYS = (
    "camera_capture_fps",
    "display_fps",
    "physics_hz",
    "capture_to_display_ms_p50",
    "capture_to_display_ms_p95",
)


def _missing_engine(profile: str) -> str | None:
    """Return why a profile cannot run truthfully, or None when it can."""
    if profile == "engine_warp_cv2":
        missing = [
            name
            for name, module in (("warp-lang", "warp"), ("opencv", "cv2"))
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            return f"missing optional dependencies: {', '.join(missing)}"
    return None


def _run_once(profile: str, extra: list[str], args: argparse.Namespace) -> dict:
    log_path = ROOT / "var" / "eval" / f"bench_e2e_{profile}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-X", "utf8", "-m", "app.main",
        "--mode", "mock",
        "--frames", str(args.frames),
        "--headless",
        "--no-calibration",
        "--geometry", args.geometry,
        "--width", str(args.width),
        "--height", str(args.height),
        "--log", str(log_path),
        *extra,
    ]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout_s,
    )
    wall_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise SystemExit(
            f"[bench-e2e] {profile} failed with exit code {proc.returncode}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-2000:]}"
        )
    start = proc.stdout.find("{")
    if start < 0:
        raise SystemExit(f"[bench-e2e] {profile}: no summary JSON on stdout")
    try:
        summary = json.loads(proc.stdout[start:])
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[bench-e2e] {profile}: summary parse failed: {exc}") from exc
    return {
        "frames": int(summary["frames"]),
        "wall_s": wall_s,
        "timing": {key: float(summary["timing"].get(key, 0.0)) for key in _TIMING_KEYS},
        "log": str(log_path.relative_to(ROOT)),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end A/B benchmark")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--geometry", default="silhouette",
                        choices=["stub", "silhouette", "pattern"])
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per profile; the archived table used 3 and took medians")
    parser.add_argument("--profiles", nargs="*", default=list(PROFILES),
                        choices=list(PROFILES))
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--out", default=str(ROOT / "var" / "eval" / "bench_e2e.json"))
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    results: dict[str, dict] = {}
    for profile in args.profiles:
        reason = _missing_engine(profile)
        if reason is not None:
            results[profile] = {"skipped": reason}
            print(f"[bench-e2e] skip {profile}: {reason}")
            continue
        runs = []
        for _ in range(args.repeats):
            runs.append(_run_once(profile, PROFILES[profile], args))
        timings = {
            key: _median([run["timing"][key] for run in runs]) for key in _TIMING_KEYS
        }
        results[profile] = {
            "runs": len(runs),
            "frames": runs[0]["frames"],
            "wall_s_median": _median([run["wall_s"] for run in runs]),
            "timing_median": timings,
            "per_run_display_fps": [round(run["timing"]["display_fps"], 2) for run in runs],
            "log": runs[0]["log"],
        }
        print(
            f"[bench-e2e] {profile}: display_fps={timings['display_fps']:.1f} "
            f"p50={timings['capture_to_display_ms_p50']:.1f} ms "
            f"p95={timings['capture_to_display_ms_p95']:.1f} ms"
        )

    payload = {
        "kind": "local_host_end_to_end_benchmark",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "protocol": (
            f"app.main --mode mock --headless --no-calibration; {args.frames} frames at "
            f"{args.width}x{args.height}; geometry={args.geometry}; medians over "
            f"{args.repeats} run(s) per profile"
        ),
        "config": {
            "frames": args.frames,
            "width": args.width,
            "height": args.height,
            "geometry": args.geometry,
            "repeats": args.repeats,
        },
        "profiles": results,
        "note": (
            "Local-host engineering evidence, NOT the SPEC §17.10 target-hardware "
            "table; measured_numbers stays empty. Host load causes ±30% spread."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench-e2e] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
