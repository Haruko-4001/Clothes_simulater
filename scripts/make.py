"""Cross-platform Makefile target runner (Windows fallback for GNU make)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GARMENT = "assets/samples/garment.png"

TARGETS = (
    "install",
    "install-engine",
    "install-perception",
    "test",
    "mock-run",
    "bootstrap-asset",
    "eval-report",
    "bench",
    "serve",
    "smoke",
)


def command_for(target: str, env: Mapping[str, str]) -> list[str]:
    """Build a target's command from the *already merged* environment.

    This must run inside ``main()``, after ``KEY=VALUE`` arguments have been
    applied to ``env``. Building the commands at import time made the
    documented ``make.py bootstrap-asset IMG=<path>`` override a silent no-op:
    ``os.environ.get("IMG")`` was evaluated before the override existed.
    """
    py = sys.executable
    if target == "install":
        return [py, "-m", "pip", "install", "-e", ".[test]"]
    if target == "install-engine":
        return [py, "-m", "pip", "install", "-e", ".[test,engine]"]
    if target == "install-perception":
        return [py, "-m", "pip", "install", "-e", ".[perception]"]
    if target == "test":
        return [py, "-m", "pytest", "tests", "-q"]
    if target == "mock-run":
        return [py, "-m", "app.main", "--mode", "mock", "--frames", "300", "--headless"]
    if target == "bootstrap-asset":
        return [
            py, "-m", "app.cli", "bootstrap-asset",
            "--image", env.get("IMG", DEFAULT_GARMENT),
        ]
    if target == "eval-report":
        return [
            py, "-m", "app.cli", "eval-report",
            "--log", "var/logs/run.jsonl", "--out", "var/eval",
        ]
    if target == "bench":
        return [py, "scripts/bench_engines.py", "--out", "var/eval/bench.json"]
    if target == "serve":
        return [py, "-m", "app.serve", "--physics-solver", "warp",
                "--render-backend", "opencv", "--geometry", "pattern"]
    if target == "smoke":
        return ["bash", "scripts/smoke.sh"]
    raise KeyError(target)


def parse_overrides(args: list[str]) -> tuple[dict[str, str], list[str]]:
    """Split ``KEY=VALUE`` overrides from arguments forwarded to the target."""
    env: dict[str, str] = {}
    extra: list[str] = []
    for arg in args:
        if "=" in arg and arg.split("=", 1)[0].replace("_", "").isalnum():
            key, value = arg.split("=", 1)
            env[key] = value
        else:
            extra.append(arg)
    return env, extra


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TARGETS:
        print(f"usage: make <{'|'.join(TARGETS)}> [KEY=VALUE ...]", file=sys.stderr)
        return 2
    target = argv[0]
    overrides, extra = parse_overrides(argv[1:])
    env = dict(os.environ)
    env.update(overrides)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(command_for(target, env) + extra, cwd=ROOT, env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = ["TARGETS", "DEFAULT_GARMENT", "command_for", "parse_overrides", "main"]
