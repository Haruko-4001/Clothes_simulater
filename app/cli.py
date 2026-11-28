"""CLI: bootstrap-asset / run / eval-report / selfcheck."""
from __future__ import annotations

import argparse
import os
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


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report, per capability, whether it can actually run on this machine.

    Product-facing: a reviewer should be able to tell a working path from a
    fallback path without reading code. Nothing here imports a heavy backend;
    it only probes availability.
    """
    import importlib
    import platform

    cfg = load_config(args.mode, args.hardware, config_dir=args.config_dir)

    def probe(module: str) -> dict:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - capability probe
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        # Prefer distribution metadata: Flask 3.1 deprecates ``__version__`` and
        # reading it emits a DeprecationWarning. Fall back for cv2/warp, whose
        # distribution names differ from their import names.
        try:
            from importlib.metadata import version as dist_version

            version = dist_version(module)
        except Exception:
            version = getattr(mod, "__version__", "?")
        return {"available": True, "version": str(version)}

    def file_present(env_or_cfg: str | None) -> bool:
        if not env_or_cfg:
            return False
        import os

        return Path(os.path.expandvars(env_or_cfg)).exists()

    api_key = cfg.get("keyframe.fashn_api_key") or os.environ.get("FASHN_API_KEY")
    vendor = REPO_ROOT / "adapters_vendor"
    warp_probe = probe("warp")

    capabilities = {
        "pipeline_A_geometry": {
            "stub": "always available",
            "silhouette": "always available (pure numpy)",
            "pattern": "always available (panels + seams, pure numpy)",
            "sewformer": {
                "vendored": (vendor / "sewformer").exists(),
                "how": "bash scripts/install_real.sh --only=sewformer",
            },
            "garmentcode_upstream": {
                "vendored": (vendor / "GarmentCode").exists(),
                "note": "upstream library not bundled; pattern.py implements the "
                        "panel/stitch data model so a design loader can be swapped in",
            },
        },
        "pipeline_B_physics": {
            "numpy_xpbd": "always available",
            "warp": warp_probe,
            # Evaluated 2026-11 (see PRODUCT.md §6.1). Neither is installed by
            # this repo; the probes only report whether the host already has it.
            "newton_vbd": {
                **probe("newton"),
                "role": "optional VBD backend (experimental, integrated; `.[newton]`)",
            },
            "pyuipc_ipc": {
                **probe("pyuipc"),
                "role": "offline IPC non-penetration reference; not a runtime backend",
            },
        },
        "pipeline_B_perception": {
            "stub_trackers": "always available (mock scene only)",
            "mediapipe": {
                **probe("mediapipe"),
                "python_ok": sys.version_info < (3, 13),
                "note": "mediapipe ships no CPython 3.13 wheels; use 3.10-3.12",
            },
            "smplx": {
                **probe("smplx"),
                "model_files": file_present(cfg.get("perception.smplx_model_path")),
            },
        },
        "pipeline_B_render": {
            "pil": "always available",
            "opencv": probe("cv2"),
        },
        "keyframe_vton": {
            "stub_composite": "always available",
            "fashn_api": {"credential": bool(api_key), "network": "https://api.fashn.ai/v1"},
            "local_vton": {"vendored": (vendor / "CatVTON").exists()},
        },
        "pipeline_C_calibration": {
            "fd_gradients": "always available",
            "warp_ad": {"available": True} if warp_probe["available"] else warp_probe,
        },
        "product_shell": {
            "flask": probe("flask"),
            "serve_entry": "app.serve:main (local development preview; no auth/TLS)",
        },
    }

    payload = {
        "spec_version": __import__("tryon").SPEC_VERSION,
        "host": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible": bool(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
        },
        "selected": {
            "mode": cfg.get("mode"),
            "physics": cfg.get("physics.solver"),
            "render": cfg.get("render.backend"),
            "geometry": cfg.get("asset.geometry"),
            "keyframe": cfg.get("keyframe.backend"),
            "grad": cfg.get("calibration.grad_backend"),
        },
        "capabilities": capabilities,
        "verdict": "OK",
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

    p = sub.add_parser("doctor", help="per-capability availability report")
    p.add_argument("--mode", default="mock")
    p.add_argument("--hardware", default=None)
    p.add_argument("--config-dir", default=None)
    p.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
