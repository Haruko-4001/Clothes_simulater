"""CLI: bootstrap-asset / run / eval-report / selfcheck."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tryon.config import load_config  # noqa: E402
from tryon.eval.report import build_report  # noqa: E402
from tryon.factory import build  # noqa: E402
from tryon.logging import dumps_json, setup_logging  # noqa: E402


def _load_image(path: str) -> np.ndarray:
    from app.main import load_garment_image

    return load_garment_image(path, load_config("mock"))


def cmd_bootstrap(args: argparse.Namespace) -> int:
    cfg = load_config(args.mode, args.hardware, config_dir=args.config_dir)
    setup_logging()
    components = build(cfg)
    builder = components["AssetBuilder"]
    image = _load_image(args.image)
    asset = builder.build(image, timeout_s=float(cfg.require("asset.offline_timeout_s")))
    payload = {
        "asset_id": asset.asset_id,
        "version": asset.version,
        "garment_class": asset.garment_class.value,
        "material_state": asset.material.state.value,
        "confidence": asset.confidence.as_dict(),
        "provenance": {k: v.value for k, v in asset.provenance.items()},
        "needs_human_review": asset.needs_human_review,
        "mesh": {
            "n_vertices": asset.rest_mesh.n_vertices,
            "n_faces": asset.rest_mesh.n_faces,
            "thickness": asset.rest_mesh.thickness,
            "is_closed": asset.rest_mesh.is_closed,
            "has_isolated_edge": asset.rest_mesh.has_isolated_edge,
            "self_intersection_count": asset.rest_mesh.self_intersection_count,
        },
        "g1": asset.sewing_pattern.get("g1", {}),
        "g2": asset.sewing_pattern.get("g2", {}),
        "material": {
            "vector": asset.material.to_vector().tolist(),
            "state": asset.material.state.value,
        },
    }
    print(dumps_json(payload, indent=2))
    return 0


def cmd_eval_report(args: argparse.Namespace) -> int:
    cfg = load_config(args.mode, args.hardware, config_dir=args.config_dir)
    report = build_report(cfg, log_path=args.log, output_dir=args.out)
    print(dumps_json(report, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from app.main import main as run_main

    argv = []
    if args.mode:
        argv += ["--mode", args.mode]
    if args.hardware:
        argv += ["--hardware", args.hardware]
    argv += ["--frames", str(args.frames), "--garment", args.garment, "--log", args.log]
    if args.no_keyframes:
        argv.append("--no-keyframes")
    return int(run_main(argv))


def cmd_selfcheck(args: argparse.Namespace) -> int:
    import numpy

    from tryon import SPEC_VERSION

    cfg = load_config("mock")
    payload = {
        "spec_version": SPEC_VERSION,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "mode": cfg.get("mode"),
        "five_fps_calibers": [
            "camera_capture_fps",
            "display_fps",
            "physics_hz",
            "keyframe_fps",
            "generation_throughput_fps",
        ],
        "status": "OK",
    }
    print(dumps_json(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tryon-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("bootstrap-asset", help="run pipeline A once")
    p.add_argument("--image", default=str(REPO_ROOT / "assets" / "samples" / "garment.png"))
    p.add_argument("--mode", default="mock")
    p.add_argument("--hardware", default=None)
    p.add_argument("--config-dir", default=None)
    p.set_defaults(func=cmd_bootstrap)

    p = sub.add_parser("eval-report", help="write evaluation report")
    p.add_argument("--log", default=str(REPO_ROOT / "var" / "logs" / "run.jsonl"))
    p.add_argument("--out", default=str(REPO_ROOT / "var" / "eval"))
    p.add_argument("--mode", default="mock")
    p.add_argument("--hardware", default=None)
    p.add_argument("--config-dir", default=None)
    p.set_defaults(func=cmd_eval_report)

    p = sub.add_parser("run", help="run the realtime mock loop")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--garment", default=str(REPO_ROOT / "assets" / "samples" / "garment.png"))
    p.add_argument("--log", default=str(REPO_ROOT / "var" / "logs" / "run.jsonl"))
    p.add_argument("--mode", default="mock")
    p.add_argument("--hardware", default=None)
    p.add_argument("--no-keyframes", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("selfcheck", help="environment self check")
    p.set_defaults(func=cmd_selfcheck)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
