"""HUD: always shows material state, framing verdict and guide text (SPEC §8).

IMPL: src/tryon/display/hud.py::HUD
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import Config
from ..types import MaterialState, RGBImage, SystemStatus


_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


class HUD:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.text: str = ""
        self._font_cache: dict[int, ImageFont.ImageFont] = {}

    # ------------------------------------------------------------------
    def _font(self, size: int):
        if size in self._font_cache:
            return self._font_cache[size]
        font = None
        for path in _FONT_CANDIDATES:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, size=size)
                    break
                except OSError:
                    continue
        if font is None:
            try:
                font = ImageFont.load_default(size=size)
            except TypeError:  # older Pillow
                font = ImageFont.load_default()
        self._font_cache[size] = font
        return font

    # ------------------------------------------------------------------
    def _lines(self, status: SystemStatus) -> list[str]:
        cfg = self.cfg
        lines: list[str] = []
        state = status.material_state
        lines.append(f"材质状态: {state.value}")
        if state != MaterialState.CALIBRATED:
            lines.append("未校准：材质为 AI 先验，禁止宣称物理已就绪")
        else:
            lines.append("材质已校准（场景标签 + TTL，非永久 SKU 参数）")

        if bool(cfg.get("hud.show_framing", True)):
            lines.append(f"取景: {status.framing.verdict.value}")
        if bool(cfg.get("hud.show_guide_text", True)):
            lines.append(f"引导: {status.framing.guide_text}")

        tau_back = float(cfg.require("asset.back_confidence_threshold"))
        if float(status.confidence.back_geometry) < tau_back:
            lines.append(
                f"背面几何: AI 补全 (置信度 {status.confidence.back_geometry:.2f})"
            )

        if bool(cfg.get("hud.show_confidence", True)):
            lines.append(
                f"置信度: overall {status.confidence.overall:.2f} / seg "
                f"{status.confidence.segmentation:.2f}"
            )
        if status.notice:
            lines.append(f"提示: {status.notice}")
        lines.append(
            "不可承诺: 首次拉扯即真实手感 / 单图确定背面 / 侧背实时换装 / 4K60+高细节"
        )
        if status.timing.capture_to_display_ms_p95 > 0:
            lines.append(f"p95 延迟: {status.timing.capture_to_display_ms_p95:.1f} ms")
        return lines

    def draw(self, frame: RGBImage, status: SystemStatus) -> RGBImage:
        arr = np.array(frame, dtype=np.uint8, copy=True)
        h, w = arr.shape[:2]
        lines = self._lines(status)
        self.text = "\n".join(lines)
        size = max(12, int(round(h / 42.0)))
        font = self._font(size)
        line_h = int(size * 1.45)
        pad = max(6, size // 2)
        panel_w = min(w, max(int(len(s) * size * 0.95) for s in lines) + 2 * pad)
        panel_h = min(h, line_h * len(lines) + 2 * pad)
        if panel_w <= 0 or panel_h <= 0:
            return arr

        # Darken only the HUD panel in numpy; a full-frame RGBA composite
        # costs ~20 ms at 1080p for no visual benefit.
        alpha = 168.0 / 255.0
        dark = np.array([10.0, 14.0, 20.0], dtype=np.float32)
        region = arr[0:panel_h, 0:panel_w].astype(np.float32)
        arr[0:panel_h, 0:panel_w] = np.clip(
            region * (1.0 - alpha) + dark * alpha, 0, 255
        ).astype(np.uint8)

        panel = Image.fromarray(arr[0:panel_h, 0:panel_w])
        draw = ImageDraw.Draw(panel)
        y = pad
        for i, line in enumerate(lines):
            colour = (235, 240, 248)
            if i == 0:
                colour = (255, 210, 120)
            elif "AI 补全" in line or "未校准" in line:
                colour = (255, 170, 140)
            draw.text((pad, y), line, font=font, fill=colour)
            y += line_h
        arr[0:panel_h, 0:panel_w] = np.asarray(panel, dtype=np.uint8)
        return arr


__all__ = ["HUD"]
