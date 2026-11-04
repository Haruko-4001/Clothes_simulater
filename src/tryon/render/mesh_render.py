"""PIL mesh renderer — physics-driven visible layer (no generation).

IMPL: src/tryon/render/mesh_render.py::PILMeshRenderer
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from ..math import project
from ..types import Frame, GarmentAsset, PhysicsState, RenderOutput


class PILMeshRenderer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._color_cache: dict[str, tuple[int, int, int]] = {}

    def render(
        self,
        frame: Frame,
        asset: GarmentAsset,
        state: PhysicsState,
        camera,
    ) -> RenderOutput:
        K = np.asarray(getattr(camera, "K"), dtype=np.float64)
        T_CW = np.asarray(getattr(camera, "T_CW"), dtype=np.float64)
        original = np.asarray(frame.image)
        h, w = original.shape[:2]
        V = np.asarray(state.positions, dtype=np.float64)
        F = asset.rest_mesh.faces

        # Draw the mesh layer at a reduced internal resolution when configured;
        # the visible output remains full camera resolution.
        scale = float(self.cfg.get("render.scale", 1.0)) if self.cfg is not None else 1.0
        scale = min(max(scale, 0.1), 1.0)
        sw, sh = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        if scale < 1.0:
            K_draw = K.copy()
            K_draw[0, 0] *= scale
            K_draw[0, 2] *= scale
            K_draw[1, 1] *= scale
            K_draw[1, 2] *= scale
        else:
            K_draw = K

        try:
            uv, depth = project(K_draw, T_CW, V)
        except Exception:
            return RenderOutput(
                image=np.array(original, copy=True),
                mask=np.zeros((h, w), dtype=np.uint8),
                coverage_bbox=(0, 0, 0, 0),
            )

        face_depth = depth[F].mean(axis=1)
        valid = np.all(depth[F] > 1.0e-4, axis=1)
        order = np.argsort(-face_depth)

        tri = V[F]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nlen = np.linalg.norm(normals, axis=1)
        nlen[nlen < 1e-12] = 1.0
        normals = normals / nlen[:, None]
        # Backface culling: halves the polygon count for the closed double
        # shell and keeps the inner surface from being painted over the outer.
        R = T_CW[:3, :3]
        cam_center = -R.T @ T_CW[:3, 3]
        centroids = tri.mean(axis=1)
        facing = np.einsum("ij,ij->i", normals, cam_center[None, :] - centroids)
        front_facing = facing > 0.0
        light = np.array([0.35, 0.80, 0.48], dtype=np.float64)
        light /= np.linalg.norm(light)
        shade = 0.55 + 0.45 * np.abs(normals @ light)
        base = np.array(self._garment_color(asset), dtype=np.float64)

        pil = Image.new("RGB", (sw, sh), (0, 0, 0)) if scale < 1.0 else Image.fromarray(original)
        draw = ImageDraw.Draw(pil)
        mask_img = Image.new("L", (sw, sh), 0)
        mdraw = ImageDraw.Draw(mask_img)

        for fi in order:
            if not bool(valid[fi]) or not bool(front_facing[fi]):
                continue
            pts = uv[F[fi]]
            if np.any(~np.isfinite(pts)):
                continue
            xy = [(float(p[0]), float(p[1])) for p in pts]
            # Cheap frustum reject.
            if max(p[0] for p in xy) < 0 or min(p[0] for p in xy) > sw:
                continue
            if max(p[1] for p in xy) < 0 or min(p[1] for p in xy) > sh:
                continue
            col = np.clip(base * shade[fi], 0, 255).astype(np.uint8)
            colour = (int(col[0]), int(col[1]), int(col[2]))
            draw.polygon(xy, fill=colour)
            mdraw.polygon(xy, fill=255)

        if scale < 1.0:
            layer = np.asarray(pil.resize((w, h), Image.BILINEAR), dtype=np.uint8)
            mask_full = mask_img.resize((w, h), Image.NEAREST)
            mask = (np.asarray(mask_full, dtype=np.uint8) > 127).astype(np.uint8)
            out = np.array(original, copy=True)
            m = mask > 0
            out[m] = layer[m]
            image = out
        else:
            mask = (np.asarray(mask_img, dtype=np.uint8) > 127).astype(np.uint8)
            image = np.asarray(pil, dtype=np.uint8)

        ys, xs = np.where(mask > 0)
        bbox = (0, 0, 0, 0)
        if xs.size:
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        return RenderOutput(image=image, mask=mask, coverage_bbox=bbox)

    # ------------------------------------------------------------------
    def _garment_color(self, asset: GarmentAsset) -> tuple[int, int, int]:
        cached = self._color_cache.get(asset.asset_id)
        if cached is not None:
            return cached
        colour = (78, 115, 158)
        appearance = asset.appearance_mesh
        src = getattr(appearance, "source_image", None)
        if src is not None:
            arr = np.asarray(src, dtype=np.float64)
            border = np.concatenate(
                [arr[0, :, :], arr[-1, :, :], arr[:, 0, :], arr[:, -1, :]], axis=0
            )
            bg = np.median(border, axis=0)
            dist = np.linalg.norm(arr - bg[None, None, :], axis=2)
            fg = arr[dist > 18.0]
            if fg.size:
                mean = fg.mean(axis=0)
                colour = (int(mean[0]), int(mean[1]), int(mean[2]))
        self._color_cache[asset.asset_id] = colour
        return colour


__all__ = ["PILMeshRenderer"]
