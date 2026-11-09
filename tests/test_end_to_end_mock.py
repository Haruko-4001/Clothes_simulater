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


def test_eval_report_wires_every_spec7_metric_without_inventing_numbers(
    mock_cfg, tmp_path
):
    """§7 / §15 Phase 5: metrics.py must be reachable from report.py.

    The target-hardware table stays null and measured_numbers stays empty;
    only caller-supplied ground truth produces numbers, under a clearly
    labelled synthetic_validation section.
    """
    from tryon.eval.report import build_report

    def reject(constant):
        raise ValueError("non-standard JSON constant: " + constant)

    log = tmp_path / "run.jsonl"
    rows = []
    for i in range(5):
        rows.append(json.dumps({
            "frame_id": i + 1,
            "pts_ms": 33.0 * i,
            "framing": {"verdict": "OK", "visible_keypoint_ratio": 1.0,
                        "guide_text": "取景合规"},
            "material_state": "AI_PRIOR",
            "physics": {"attached_count": 0, "penetration_count": 0,
                        "energy": 1.0 - 0.01 * i},
            "timing": {
                "camera_capture_fps": 30, "display_fps": 30, "physics_hz": 60,
                "keyframe_fps": 0.5, "generation_throughput_fps": 2.0,
                "capture_to_display_ms_p50": 20,
                "capture_to_display_ms_p95": 40,
                "keyframe_latency_ms_p95": 100,
                "capture_to_display_ms_max": 55,
            },
            "events": [],
        }))
    log.write_text("\n".join(rows), encoding="utf-8")

    eye3 = np.eye(3, dtype=np.float64)
    report = build_report(
        mock_cfg,
        log_path=log,
        output_dir=tmp_path / "eval",
        n_vertices=10,
        latency_samples_ms=[10.0, 20.0, 30.0, 40.0, 100.0],
        ground_truth={
            "mask_pred": np.array([[1, 0], [0, 1]], dtype=np.uint8),
            "mask_true": np.array([[1, 0], [0, 0]], dtype=np.uint8),
            "points_pred": eye3,
            "points_true": eye3,
            "normals_pred": eye3,
            "normals_true": eye3,
            "trajectory_pred": eye3,
            "trajectory_true": eye3,
            "horizon": 3,
            "theta_hat": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            "theta_star": np.ones(7),
            "penetration_counts": [0, 0, 0, 0, 0],
            "n_vertices": 10,
            "energies": [1.0, 0.99, 0.98],
            "gamma": 0.15,
            "dt": 1.0 / 30.0,
            # §7 convergence: 5 strictly decreasing windows, each <2% change.
            "variances": [1.0, 0.995, 0.990, 0.985, 0.980],
            "latency_samples_ms": [10.0, 20.0, 30.0, 40.0, 100.0],
        },
    )

    # Framework counters really come from metrics.py.
    assert report["mock_run_counters"]["framing_compliance_rate"] == 1.0
    assert report["mock_run_counters"]["degrade_event_count"] == 0
    assert report["mock_run_counters"]["energy_decay_ok"] is True
    assert report["mock_run_counters"]["latency_p95_ms"] > \
        report["mock_run_counters"]["latency_p50_ms"]
    assert report["mock_run_counters"]["penetration_rate"] == 0.0
    assert "generation_throughput_fps" not in report["mock_run_counters"]["timing_final"]

    # Every §7 family has a named implementation, and none is faked.
    assert len(report["metric_families"]["symbols"]) == 11
    assert all(s["status"] == "AVAILABLE" for s in report["metric_families"]["symbols"])
    synthetic = report["synthetic_validation"]
    assert synthetic["iou"] == 0.5
    assert synthetic["chamfer_distance"] == 0.0
    assert synthetic["normal_error"] == 0.0
    assert synthetic["ade"] == 0.0 and synthetic["fde"] == 0.0
    assert synthetic["parameter_error"] > 0.0
    assert synthetic["convergence"] is True
    assert synthetic["energy_decay_ok"] is True

    assert report["measured_numbers"] == []
    assert all(
        row["value"] is None and row["status"] == "NOT_MEASURED"
        for row in report["target_hardware_measurements"]["rows"]
    )
    written = (tmp_path / "eval" / "report.json").read_text(encoding="utf-8")
    parsed = json.loads(written, parse_constant=reject)
    assert parsed["measured_numbers"] == []


def test_mock_loop_is_deterministic_when_calibration_is_disabled(
    mock_cfg, garment_image, tmp_path
):
    """Physics/framing are pure functions of the seed; only pipeline C timing varies.

    The background calibration worker is intentionally asynchronous (SPEC §5),
    so its submission count is wall-clock dependent. Disabling it isolates the
    deterministic part and documents where the nondeterminism comes from.
    """
    def run(tag):
        cfg = mock_cfg.clone()
        cfg.set("camera.width", 160)
        cfg.set("camera.height", 120)
        cfg.set("asset.mesh_resolution", [10, 7])
        log = tmp_path / f"{tag}.jsonl"
        runner = MockRunner(
            cfg,
            garment_image=garment_image,
            log_path=log,
            headless=True,
            seed=7,
            enable_keyframes=False,
            enable_calibration=False,
        )
        summary = runner.run(60)
        rows = [
            json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return summary, rows

    s1, r1 = run("a")
    s2, r2 = run("b")
    assert s1["framing_histogram"] == s2["framing_histogram"]
    assert s1["material_histogram"] == s2["material_histogram"]
    assert [r["framing"]["verdict"] for r in r1] == [r["framing"]["verdict"] for r in r2]
    assert [r["physics"]["energy"] for r in r1] == [r["physics"]["energy"] for r in r2]
    assert [r["physics"]["penetration_count"] for r in r1] == [
        r["physics"]["penetration_count"] for r in r2
    ]
