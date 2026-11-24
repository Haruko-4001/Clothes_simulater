"""Engine benchmark: numpy XPBD vs Warp XPBD, PIL vs OpenCV rasterisation.

Reproduces the local-host numbers quoted in README ("本机实测基准").
Writes JSON to ``--out`` (default ``var/eval/bench.json``, git-ignored).

Scope: this measures the OPTIONAL engine backends on the host it runs on.
It is not the SPEC §17.10 target-hardware table and must never be reported as
one; ``STRUCTURED_REPORT.json`` keeps ``measured_numbers`` empty.

Usage:
    python scripts/bench_engines.py
    python scripts/bench_engines.py --out var/eval/bench.json --steps 120
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tryon.asset.asset_builder import AssetBuilder  # noqa: E402
from tryon.asset.drape import DrapeAligner  # noqa: E402
from tryon.asset.material_prior import MaterialPrior  # noqa: E402
from tryon.asset.mesh_repair import MeshRepairer  # noqa: E402
from tryon.asset.registry import AssetRegistry  # noqa: E402
from tryon.asset.stubs import StubGeometryEstimator, StubSegmenter  # noqa: E402
from tryon.config import load_config  # noqa: E402
from tryon.physics.solver import PhysicsSolver  # noqa: E402

# (56, 52) is the largest loft that fits physics.max_vertices = 6000
# (2 * nu * (nv + 1) = 5936 vertices); it is the SPEC §8 budget point.
GRIDS = [(12, 9), (16, 12), (28, 20), (40, 35), (56, 52)]


def _probe(module: str) -> str | None:
    try:
        mod = __import__(module)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return None


def _synthetic_garment() -> np.ndarray:
    img = np.zeros((480, 384, 3), dtype=np.uint8)
    img[:, :] = (240, 240, 240)
    img[60:440, 80:300] = (80, 110, 160)
    return img


def _build_asset(cfg, resolution: tuple[int, int]):
    local = cfg.clone()
    local.set("asset.mesh_resolution", [resolution[0], resolution[1]])
    builder = AssetBuilder(
        local,
        StubSegmenter(local),
        StubGeometryEstimator(local),
        MeshRepairer(local),
        DrapeAligner(local),
        MaterialPrior(local),
        AssetRegistry(local),
    )
    return builder.build(_synthetic_garment(), timeout_s=120.0)


def _time_steps(solver_factory: Callable, cfg, asset, steps: int, warmup: int,
                repeat: int = 3) -> dict[str, Any]:
    """Median of ``repeat`` runs; each run rebuilds the solver (cold caches).

    A laptop GPU shared with the desktop produces large run-to-run variance,
    so a single sample is not evidence. ``min``/``max`` are kept so the spread
    is visible instead of being hidden behind one flattering number.
    """
    rates: list[float] = []
    last_state = None
    for _ in range(max(1, repeat)):
        solver = solver_factory(cfg)
        state = solver.initial_state(asset)
        dt = float(cfg.require("physics.dt"))
        for _ in range(warmup):
            state = solver.step(state, asset, [], dt)
        t0 = time.perf_counter()
        for _ in range(steps):
            state = solver.step(state, asset, [], dt)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        rates.append(steps / elapsed)
        last_state = state
    median = float(np.median(rates))
    return {
        "steps_per_s_median": round(median, 1),
        "steps_per_s_min": round(min(rates), 1),
        "steps_per_s_max": round(max(rates), 1),
        "samples": len(rates),
        "ms_per_step_median": round(1000.0 / median, 3) if median > 0 else None,
        "finite": bool(last_state is not None and np.all(np.isfinite(last_state.positions))),
        "energy_j": round(float(last_state.energy), 6) if last_state is not None else None,
    }


def _time_render(renderer_cls, cfg, asset, steps: int) -> dict[str, float] | None:
    try:
        renderer = renderer_cls(cfg)
    except ImportError:
        return None
    from tryon.perception.stubs import StubCamera

    camera = StubCamera(cfg)
    solver = PhysicsSolver(cfg)
    state = solver.initial_state(asset)
    dt = float(cfg.require("physics.dt"))
    for _ in range(10):
        state = solver.step(state, asset, [], dt)
    frame = camera.capture()
    for _ in range(3):
        renderer.render(frame, asset, state, camera)
    t0 = time.perf_counter()
    for _ in range(steps):
        renderer.render(frame, asset, state, camera)
    return {"ms_per_frame": round((time.perf_counter() - t0) / steps * 1000.0, 2)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Engine backend benchmark (local host only)")
    parser.add_argument("--out", default=str(ROOT / "var" / "eval" / "bench.json"))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--substeps", type=int, default=None,
                        help="override physics.substeps (solver tuning sweep)")
    parser.add_argument("--iters", type=int, default=None,
                        help="override physics.xpbd_iters (solver tuning sweep)")
    args = parser.parse_args(argv)

    cfg = load_config("mock", None)
    cfg.set("camera.width", int(args.width))
    cfg.set("camera.height", int(args.height))
    if args.substeps is not None:
        cfg.set("physics.substeps", int(args.substeps))
    if args.iters is not None:
        cfg.set("physics.xpbd_iters", int(args.iters))

    warp_version = _probe("warp")
    cv2_version = _probe("cv2")

    physics: list[dict[str, Any]] = []
    for grid in GRIDS:
        asset = _build_asset(cfg, grid)
        entry: dict[str, Any] = {
            "grid": list(grid),
            "vertices": asset.rest_mesh.n_vertices,
            "faces": asset.rest_mesh.n_faces,
            "numpy": _time_steps(PhysicsSolver, cfg, asset, args.steps, args.warmup, args.repeat),
        }
        if warp_version is not None:
            try:
                from tryon.physics.warp_solver import WarpSolver

                entry["warp"] = _time_steps(
                    WarpSolver, cfg, asset, args.steps, args.warmup, args.repeat
                )
            except Exception as exc:  # pragma: no cover - host dependent
                entry["warp"] = {"error": f"{type(exc).__name__}: {exc}"}
        physics.append(entry)
        w = entry.get("warp", {})
        print(
            f"grid {grid} verts={entry['vertices']:>5} "
            f"numpy={entry['numpy']['steps_per_s_median']:>7} "
            f"[{entry['numpy']['steps_per_s_min']}-{entry['numpy']['steps_per_s_max']}] "
            f"warp={w.get('steps_per_s_median', 'n/a')} "
            f"[{w.get('steps_per_s_min', '-')}-{w.get('steps_per_s_max', '-')}] steps/s"
        )

    render: dict[str, Any] = {}
    asset = _build_asset(cfg, (28, 20))
    from tryon.render.mesh_render import PILMeshRenderer

    render["pil"] = _time_render(PILMeshRenderer, cfg, asset, 20)
    if cv2_version is not None:
        from tryon.render.cv2_render import OpenCVMeshRenderer

        render["opencv"] = _time_render(OpenCVMeshRenderer, cfg, asset, 20)
    print(f"render {args.width}x{args.height} @{asset.rest_mesh.n_faces} faces: {render}")

    payload = {
        "kind": "local_host_engine_benchmark",
        "disclaimer": (
            "Local-host engineering evidence only. NOT the SPEC §17.10 "
            "target-hardware table; measured_numbers in STRUCTURED_REPORT stays empty."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reproduce": "python scripts/bench_engines.py --out var/eval/bench.json",
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp_lang": warp_version,
            "opencv": cv2_version,
        },
        "protocol": {
            "physics_steps": args.steps,
            "warmup_steps": args.warmup,
            "repeats": args.repeat,
            "statistic": "median of repeats; min/max kept to expose host variance",
            "frame_size": [args.width, args.height],
            "note": "physics = fixed-dt 1/60 s steps in a tight loop after warmup",
        },
        "physics": physics,
        "render": render,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
