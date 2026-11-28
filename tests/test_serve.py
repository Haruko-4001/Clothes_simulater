"""app/serve.py HTTP surface: Flask test client, no sockets or background thread."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app import serve


class _Cfg(dict):
    """Minimal Config stand-in: create_app only calls .get(dotted_key)."""


class FakeSession:
    def __init__(self, jpeg: bytes | None = b"\xff\xd8fake-jpeg") -> None:
        self.cfg = _Cfg(
            {
                "mode": "mock",
                "physics.solver": "stub_xpbd",
                "render.backend": "stub",
                "asset.geometry": "pattern",
                "keyframe.backend": "stub",
                "calibration.grad_backend": "fd",
            }
        )
        self.jpeg = jpeg
        self.swapped: list[np.ndarray] = []

    def status(self) -> dict:
        return {"ready": True, "frame_id": 7, "serve": {"swapping": False}}

    def latest_jpeg(self) -> bytes | None:
        return self.jpeg

    def swap(self, image: np.ndarray) -> dict:
        self.swapped.append(np.asarray(image))
        return {
            "asset_id": "g_test",
            "version": 0,
            "g1_pass": True,
            "mesh": {"n_vertices": 624, "n_faces": 1248},
        }

    def active_backends(self) -> dict[str, str]:
        return {"physics": "FakePhysics", "geometry": "FakeGeometry"}


def _png(width: int = 64, height: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (30, 60, 90)).save(buf, format="PNG")
    return buf.getvalue()


def _client(session: FakeSession):
    app = serve.create_app(session)
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture()
def client_and_session():
    session = FakeSession()
    return _client(session), session


def test_index_serves_the_single_page_shell(client_and_session):
    client, _ = client_and_session
    response = client.get("/")
    assert response.status_code == 200
    assert "实时布料试穿" in response.get_data(as_text=True)


def test_config_reports_configured_and_active_backends(client_and_session):
    client, _ = client_and_session
    payload = client.get("/api/config").get_json()
    assert payload["physics"] == "stub_xpbd"
    assert payload["active"]["physics"] == "FakePhysics"
    assert payload["dev_only"] is True


def test_status_is_the_session_payload(client_and_session):
    client, _ = client_and_session
    payload = client.get("/api/status").get_json()
    assert payload["ready"] is True
    assert payload["frame_id"] == 7


def test_frame_jpeg_and_503_before_the_first_frame():
    session = FakeSession()
    assert _client(session).get("/api/frame.jpg").status_code == 200

    empty = FakeSession(jpeg=None)
    assert _client(empty).get("/api/frame.jpg").status_code == 503


def test_upload_without_a_part_is_400(client_and_session):
    client, _ = client_and_session
    response = client.post("/api/garment", data={}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_upload_rejects_non_image_content_type(client_and_session):
    client, _ = client_and_session
    response = client.post(
        "/api/garment",
        data={"image": (io.BytesIO(b"not-an-image"), "note.txt", "text/plain")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 415


def test_upload_rejects_tiny_images(client_and_session):
    client, _ = client_and_session
    response = client.post(
        "/api/garment",
        data={"image": (io.BytesIO(_png(16, 16)), "g.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400


def test_upload_rejects_images_over_the_pixel_budget(client_and_session, monkeypatch):
    client, _ = client_and_session
    monkeypatch.setattr(serve, "MAX_UPLOAD_PIXELS", 100)
    response = client.post(
        "/api/garment",
        data={"image": (io.BytesIO(_png(64, 64)), "g.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413


def test_upload_rebuilds_the_asset_and_returns_its_summary(client_and_session):
    client, session = client_and_session
    response = client.post(
        "/api/garment",
        data={"image": (io.BytesIO(_png(64, 64)), "g.png", "image/png")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["asset"]["asset_id"] == "g_test"
    assert len(session.swapped) == 1
    assert session.swapped[0].shape == (64, 64, 3)


def test_parse_args_matches_main_defaults_and_accepts_backend_flags():
    args = serve.parse_args([])
    # None, not "mock": configs/real.yaml must not be silently overridden.
    assert args.mode is None
    args = serve.parse_args(
        [
            "--mode", "real",
            "--keyframe-backend", "fashn",
            "--grad-backend", "warp",
            "--self-collision",
        ]
    )
    assert args.mode == "real"
    assert args.keyframe_backend == "fashn"
    assert args.grad_backend == "warp"
    assert args.self_collision is True
