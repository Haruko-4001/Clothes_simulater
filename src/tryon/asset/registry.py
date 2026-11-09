"""Garment asset registry: version / confidence / TTL / scene_tag.

IMPL: src/tryon/asset/registry.py
COMMIT is the only operation that increments ``version`` (SPEC §9.2 / §5).
Calibrated parameters are never written permanently into a SKU; they carry a
``scene_tag`` and a TTL.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config, REPO_ROOT
from ..logging import dumps_json
from ..types import Confidence, GarmentAsset, MaterialParams, MaterialState


def asset_id_for(image_bytes: bytes) -> str:
    return "g_" + hashlib.sha1(image_bytes).hexdigest()[:8]


def _material_to_dict(m: MaterialParams) -> dict[str, Any]:
    return {
        "vector": m.to_vector().tolist(),
        "state": m.state.value,
        "variance": dict(m.variance),
        "scene_tag": m.scene_tag,
        "calibrated_at": m.calibrated_at,
        "ttl_s": m.ttl_s,
    }


class AssetRegistry:
    """JSON-backed metadata registry. Mesh payloads stay in memory."""

    def __init__(
        self,
        cfg: Config | None = None,
        path: str | Path | None = None,
    ) -> None:
        self.cfg = cfg
        if path is None:
            path = REPO_ROOT / "var" / "assets" / "registry.json"
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict[str, Any]] = {}
        self._assets: dict[str, GarmentAsset] = {}
        self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if isinstance(data, dict):
            self._meta = {str(k): v for k, v in data.items() if isinstance(v, dict)}

    def flush(self) -> None:
        self.path.write_text(dumps_json(self._meta, indent=2), encoding="utf-8")

    # -- API ------------------------------------------------------------
    def register(self, asset: GarmentAsset) -> GarmentAsset:
        self._assets[asset.asset_id] = asset
        previous_version = int(self._meta.get(asset.asset_id, {}).get("version", -1))
        if asset.version < previous_version:
            raise ValueError(
                f"version regression for {asset.asset_id}: "
                f"{asset.version} < {previous_version}"
            )
        self._meta[asset.asset_id] = self._serialise(asset)
        self.flush()
        return asset

    def get(self, asset_id: str) -> GarmentAsset | None:
        return self._assets.get(asset_id)

    def metadata(self, asset_id: str) -> dict[str, Any] | None:
        return self._meta.get(asset_id)

    def commit(
        self,
        asset: GarmentAsset,
        material: MaterialParams,
        *,
        scene_tag: str,
        now_s: float | None = None,
        ttl_s: float | None = None,
    ) -> GarmentAsset:
        """G3 COMMIT: the only version increment path (guarded by caller)."""
        now_s = time.time() if now_s is None else float(now_s)
        ttl = material.ttl_s if ttl_s is None else float(ttl_s)
        committed = dataclasses.replace(
            material,
            state=MaterialState.CALIBRATED,
            calibrated_at=now_s,
            scene_tag=scene_tag,
            ttl_s=ttl,
        )
        new_asset = dataclasses.replace(
            asset,
            version=asset.version + 1,
            material=committed,
            created_at=asset.created_at,
        )
        self.register(new_asset)
        return new_asset

    def version_of(self, asset_id: str) -> int:
        return int(self._meta.get(asset_id, {}).get("version", -1))

    def is_expired(self, asset_id: str, now_s: float | None = None) -> bool:
        meta = self._meta.get(asset_id)
        if not meta:
            return True
        mat = meta.get("material", {})
        calibrated_at = mat.get("calibrated_at")
        ttl = float(mat.get("ttl_s", 0.0))
        if calibrated_at is None:
            return True
        now_s = time.time() if now_s is None else float(now_s)
        return (now_s - float(calibrated_at)) > ttl

    def _serialise(self, asset: GarmentAsset) -> dict[str, Any]:
        return {
            "asset_id": asset.asset_id,
            "version": int(asset.version),
            "garment_class": asset.garment_class.value,
            "created_at": float(asset.created_at),
            "confidence": asset.confidence.as_dict(),
            "provenance": {k: v.value for k, v in asset.provenance.items()},
            "needs_human_review": bool(asset.needs_human_review),
            "material": _material_to_dict(asset.material),
            "mesh": {
                "n_vertices": int(asset.rest_mesh.n_vertices),
                "n_faces": int(asset.rest_mesh.n_faces),
                "thickness": float(asset.rest_mesh.thickness),
                "is_closed": bool(asset.rest_mesh.is_closed),
            },
        }


__all__ = ["AssetRegistry", "asset_id_for"]
