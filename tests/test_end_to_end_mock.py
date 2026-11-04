import json

import numpy as np

from app.main import MockRunner


def test_mock_end_to_end_300_frames_without_gpu_network_weights_gui(
    mock_cfg, garment_image, tmp_path
):
    cfg = mock_cfg.clone()
    cfg.set("camera.width", 160)
    cfg.set("camera.height", 120)
    cfg.set("asset.mesh_resolution", [10, 7])
    log_path = tmp_path / "run.jsonl"
    runner = MockRunner(
        cfg,
        garment_image=garment_image,
        log_path=log_path,
        headless=True,
        enable_keyframes=True,
        enable_calibration=False,
    )
    summary = runner.run(300)

    assert summary["frames"] == 300
    assert runner.display.frames_shown == 300
    assert len(runner.hud_texts) == 300
    for text in runner.hud_texts:
        assert "材质状态:" in text
        assert "取景:" in text
        assert "引导:" in text
        assert "AI 补全" in text
        assert "不可承诺" in text

    timing = summary["timing"]
    assert timing["camera_capture_fps"] > 0
    assert timing["display_fps"] > 0
    assert timing["physics_hz"] > 0
    assert timing["keyframe_fps"] > 0
    assert timing["generation_throughput_fps"] > 0
    assert timing["capture_to_display_ms_p95"] <= 100.0
    assert summary["framing_histogram"].get("OK", 0) > 0

    asset = summary["asset"]
    assert asset["g1_pass"] is True
    assert asset["material_state"] == "AI_PRIOR"

    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 300
    required = {
        "frame_id", "pts_ms", "asset_id", "asset_version", "timing",
        "framing", "material_state", "confidence", "physics", "events",
    }
    for rec in records:
        assert required.issubset(rec.keys())
        assert rec["framing"]["guide_text"] != ""
        assert len(rec["timing"]) == 9
        assert rec["physics"]["penetration_count"] == 0
        assert np.isfinite(rec["physics"]["energy"])
