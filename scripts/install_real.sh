#!/usr/bin/env bash
# Optional real-adapter installer. Nothing here is required for mock mode.
# Usage: scripts/install_real.sh --only=sewformer|fashn_vton|omnitry|catvton|wilor|diffcloth|physgaussian|engine
#
# `--only=engine` installs the P0 local engine backends (warp-lang + opencv)
# from the declared extras; it needs no vendored checkout and no credentials.
# `--only=perception` installs the P1 MediaPipe trackers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/adapters_vendor"
mkdir -p "$VENDOR"

ONLY=""
for arg in "$@"; do
  case "$arg" in
    --only=*) ONLY="${arg#--only=}" ;;
    -h|--help)
      echo "usage: $0 --only=<adapter>"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "$ONLY" ]]; then
  echo "Refusing to install everything at once. Pass --only=<adapter>." >&2
  echo "Adapters: sewformer fashn_vton omnitry catvton wilor diffcloth physgaussian engine perception" >&2
  exit 2
fi

clone() {
  local name="$1" url="$2"
  if [[ -d "$VENDOR/$name/.git" ]]; then
    echo "[skip] $name already present"
  else
    echo "[clone] $name <- $url"
    git clone --depth 1 "$url" "$VENDOR/$name"
  fi
}

case "$ONLY" in
  sewformer)
    clone sewformer https://github.com/sail-sg/sewformer
    echo "TODO: wire SewFormer Maya/fabric pipeline; license must be reviewed (SPEC §13)."
    ;;
  fashn_vton)
    clone fashn-vton-1.5 https://github.com/fashn-AI/fashn-vton-1.5
    echo "FASHN VTON v1.5: Apache-2.0 (weights terms separately reviewed)."
    ;;
  omnitry)
    clone OmniTry https://github.com/random2fly/OmniTry
    echo "OmniTry: 12 wearable classes, >=28GB; image-level only, not streaming (D11)."
    ;;
  catvton)
    clone CatVTON https://github.com/Zheng-Chong/CatVTON
    echo "CatVTON weights: CC BY-NC-SA 4.0 — research only; policy gate blocks real mode."
    ;;
  wilor)
    clone WiLoR https://github.com/rolpotamias/WiLoR
    echo "WiLoR: CC-BY-NC-ND + MANO — research only; policy gate blocks real mode."
    ;;
  diffcloth)
    clone DiffCloth https://github.com/omegaiota/DiffCloth
    echo "DiffCloth: MIT repo; CMake build and CUDA toolchain required (TODO)."
    ;;
  physgaussian)
    clone PhysGaussian https://github.com/XPandora/PhysGaussian
    echo "PhysGaussian: MPM + 3DGS; render layer only (TODO)."
    ;;
  engine)
    echo "[pip] installing local engine backends (warp-lang + opencv)"
    python -m pip install -e "$ROOT[engine]"
    echo "Engine backends installed. Select with --physics-solver warp / --render-backend opencv."
    ;;
  perception)
    echo "[pip] installing MediaPipe trackers"
    python -m pip install -e "$ROOT[perception]"
    echo "Perception installed. Requires mode: real|hybrid; falls back to Stub otherwise."
    ;;
  *)
    echo "unknown adapter: $ONLY" >&2; exit 2 ;;
esac

echo "NOTE: Dress-1-to-3 and TryOnCrafter have no published code; they are design references only."
echo "NOTE: no weight URLs are fabricated here; see scripts/fetch_weights.sh (requires network)."
