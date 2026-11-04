"""Framing compliance gate — four verdicts, hard downstream guard (SPEC §10.4).

IMPL: src/tryon/perception/framing.py::FramingGate
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..types import BodyState, Frame, FramingCompliance, FramingVerdict


GUIDE_OK = "取景合规：全身正面入镜"
GUIDE_PARTIAL = "仅渲染可见区域；画面外区域不做外推"
GUIDE_SIDE = "请正对摄像头，或等待异步 3D 渲染（侧背不提交 2D 换装）"
GUIDE_REJECT = "请调整取景：避免遮挡、透明、多层或非标准穿法"


class FramingGate:
    """Pure-algorithm gate (no external dependency; same in mock/real)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.last: FramingCompliance | None = None

    # ------------------------------------------------------------------
    def check(self, frame: Frame, body: BodyState) -> FramingCompliance:
        h, w = frame.image.shape[:2]
        enabled = bool(self.cfg.get("framing.enabled", True))
        tau_vis = float(self.cfg.require("framing.min_visible_keypoint_ratio"))
        tau_occ = float(self.cfg.require("framing.max_occlusion_ratio"))

        kp2d = self._keypoints_2d(body)
        if kp2d is None or len(kp2d) == 0:
            result = FramingCompliance(
                verdict=FramingVerdict.GUIDE,
                visible_keypoint_ratio=0.0,
                bbox_clipped=True,
                facing_estimate="unknown",
                occlusion_ratio=0.0,
                framing_confidence=0.0,
                guide_text=GUIDE_SIDE,
            )
            self.last = result
            return result

        inside = (
            (kp2d[:, 0] >= 0)
            & (kp2d[:, 0] <= w - 1)
            & (kp2d[:, 1] >= 0)
            & (kp2d[:, 1] <= h - 1)
        )
        ratio = float(inside.mean()) if len(inside) else 0.0

        x0, y0, x1, y1 = body.bbox
        bbox_clipped = bool(x0 < 0 or y0 < 0 or x1 > w or y1 > h)

        facing = str(body.smpl_params.get("facing", "frontal"))
        occlusion = float(body.smpl_params.get("occlusion", 0.0))
        multi_person = bool(body.smpl_params.get("multi_person", False))
        transparent = bool(body.smpl_params.get("transparent", False))
        multi_layer = bool(body.smpl_params.get("multi_layer", False))
        nonstandard = bool(body.smpl_params.get("nonstandard_wear", False))

        if not enabled:
            verdict = FramingVerdict.OK
            guide = GUIDE_OK
        elif occlusion > tau_occ or multi_person or transparent or multi_layer or nonstandard:
            verdict = FramingVerdict.REJECT
            guide = GUIDE_REJECT
        elif bbox_clipped or ratio < tau_vis:
            verdict = FramingVerdict.PARTIAL
            guide = GUIDE_PARTIAL
        elif facing in {"side", "back", "unknown"}:
            if bool(self.cfg.get("framing.allow_side_back", False)):
                verdict = FramingVerdict.OK
                guide = GUIDE_OK
            else:
                verdict = FramingVerdict.GUIDE
                guide = GUIDE_SIDE
        else:
            verdict = FramingVerdict.OK
            guide = GUIDE_OK

        confidence = float(
            np.clip(
                0.35 * body.confidence
                + 0.45 * ratio
                + 0.20 * (1.0 - min(occlusion / max(tau_occ, 1e-9), 1.0)),
                0.0,
                1.0,
            )
        )
        result = FramingCompliance(
            verdict=verdict,
            visible_keypoint_ratio=ratio,
            bbox_clipped=bbox_clipped,
            facing_estimate=facing,
            occlusion_ratio=float(np.clip(occlusion, 0.0, 1.0)),
            framing_confidence=confidence,
            guide_text=guide,
        )
        self.last = result
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _keypoints_2d(body: BodyState) -> np.ndarray | None:
        kp2d = body.smpl_params.get("keypoints_2d")
        if kp2d is not None:
            a = np.asarray(kp2d, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 2:
                return a[:, :2]
        kp = body.pose_keypoints
        if kp.ndim == 2 and kp.shape[1] == 2:
            return kp
        return None


__all__ = [
    "FramingGate",
    "GUIDE_OK",
    "GUIDE_PARTIAL",
    "GUIDE_SIDE",
    "GUIDE_REJECT",
]
