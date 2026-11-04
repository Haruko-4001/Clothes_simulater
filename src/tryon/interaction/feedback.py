"""Actionable user feedback strings; never silent failures (SPEC §17.16)."""
from __future__ import annotations

from ..types import FramingVerdict, GateVerdict, MaterialState


def material_feedback(state: MaterialState) -> str:
    if state == MaterialState.AI_PRIOR:
        return "材质为 AI 先验，未校准；请拉住衣物做一次完整拉放"
    if state == MaterialState.CALIBRATING:
        return "正在根据交互估计材质，请继续缓慢拉放"
    if state == MaterialState.CALIBRATED:
        return "材质已在本场景校准（有场景标签与有效期）"
    return "材质估计已降级，保持先验；请重新完整拉放或重新取景"


def framing_feedback(verdict: FramingVerdict) -> str:
    if verdict == FramingVerdict.OK:
        return "取景合规"
    if verdict == FramingVerdict.PARTIAL:
        return "仅渲染可见区域；请后退至全身入镜"
    if verdict == FramingVerdict.GUIDE:
        return "请正对摄像头；侧背请使用异步 3D 渲染"
    return "取景被拒绝：请移除遮挡、避免透明/多层衣物"


def gate_feedback(verdict: GateVerdict) -> str:
    if verdict == GateVerdict.COMMIT:
        return "材质辨识通过三闸门"
    if verdict == GateVerdict.KEEP_PRIOR:
        return "激励不足或未过闸门，保持先验"
    if verdict == GateVerdict.DEGRADE:
        return "多次交互仍未收敛，已降级"
    return "继续采集激励"


__all__ = ["material_feedback", "framing_feedback", "gate_feedback"]
