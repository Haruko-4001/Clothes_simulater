"""Cross-platform Makefile target runner (Windows fallback for GNU make)."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "install": [sys.executable, "-m", "pip", "install", "-e", ".[test]"],
    "test": [sys.executable, "-m", "pytest", "tests", "-q"],
    "mock-run": [sys.executable, "-m", "app.main", "--mode", "mock", "--frames", "300", "--headless"],
    "bootstrap-asset": [
        sys.executable,
        "-m",
        "app.cli",
        "bootstrap-asset",
        "--image",
        os.environ.get("IMG", "assets/samples/garment.png"),
    ],
    "eval-report": [
        sys.executable,
        "-m",
        "app.cli",
        "eval-report",
        "--log",
        "var/logs/run.jsonl",
        "--out",
        "var/eval",
    ],
    "smoke": ["bash", "scripts/smoke.sh"],
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TARGETS:
        print(f"usage: make <{ '|'.join(TARGETS) }> [KEY=VALUE ...]", file=sys.stderr)
        return 2
    env = dict(os.environ)
    extra: list[str] = []
    for arg in argv[1:]:
        if "=" in arg and arg.split("=", 1)[0].replace("_", "").isalnum():
            key, value = arg.split("=", 1)
            env[key] = value
        else:
            extra.append(arg)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(TARGETS[argv[0]] + extra, cwd=ROOT, env=env)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
