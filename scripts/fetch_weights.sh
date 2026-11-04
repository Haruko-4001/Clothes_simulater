#!/usr/bin/env bash
# Optional weight fetcher. Not required for mock mode; no URLs are invented.
# Requires network and a HuggingFace token for gated repos.
set -euo pipefail
echo "This script intentionally contains no fabricated URLs."
echo "Use the official model card of each adapter and a local cache path."
echo ""
echo "Examples (only after reviewing the license):"
echo "  huggingface-cli download liulj/sewformer --local-dir adapters_vendor/weights/sewformer"
echo "  # FASHN VTON v1.5: follow https://github.com/fashn-AI/fashn-vton-1.5 (Apache-2.0)"
echo "  # SMPL-X / MANO: registration required, research-only, never shipped."
echo ""
echo "No path is marked as required by configs/real.yaml."
