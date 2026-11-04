import numpy as np

from tryon.logging import JsonlFrameLogger
from tryon.math import empirical_p50, empirical_p95
from tryon.types import TimingBook


def test_five_fps_fields_never_cross_contaminate():
    t = TimingBook()
    t.camera_capture_fps = 30.0
    t.display_fps = 29.0
    t.physics_hz = 60.0
    t.keyframe_fps = 0.5
    t.generation_throughput_fps = 4.0
    snapshot = t.as_dict()
    t.physics_hz = 240.0
    assert t.camera_capture_fps == snapshot["camera_capture_fps"]
    assert t.display_fps == snapshot["display_fps"]
    assert t.keyframe_fps == snapshot["keyframe_fps"]
    assert t.generation_throughput_fps == snapshot["generation_throughput_fps"]
    # The five fps meters are independent; the 9-field log adds max only.
    assert set(t.as_dict()) != {"camera_capture_fps"}


def test_p50_p95_from_pts_ms_not_mean():
    latencies = [10.0, 12.0, 14.0, 16.0, 100.0]
    p50 = empirical_p50(latencies)
    p95 = empirical_p95(latencies)
    assert p50 != np.mean(latencies)
    assert p95 > p50
    assert p95 == float(np.percentile(np.array(latencies), 95))
    assert p95 != float(np.mean(latencies))


def test_frame_log_schema_has_nine_timing_fields(tmp_path):
    logger = JsonlFrameLogger(tmp_path / "log.jsonl")
    timing = TimingBook(
        camera_capture_fps=30,
        display_fps=29,
        physics_hz=60,
        keyframe_fps=0.5,
        generation_throughput_fps=2.0,
        capture_to_display_ms_p50=20,
        capture_to_display_ms_p95=40,
        keyframe_latency_ms_p95=100,
    ).as_dict()
    timing["capture_to_display_ms_max"] = 55.0
    record = logger.make_record(
        frame_id=1,
        pts_ms=10.0,
        asset_id="g",
        asset_version=0,
        timing=timing,
        framing_verdict="OK",
        visible_keypoint_ratio=1.0,
        guide_text="取景合规",
        material_state="AI_PRIOR",
        back_geometry_confidence=0.4,
        attached_count=1,
        penetration_count=0,
        energy=0.1,
    )
    logger.write(record)
    logger.close()
    assert len(record["timing"]) == 9
    assert record["timing"]["generation_throughput_fps"] == 2.0
    lines = (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
