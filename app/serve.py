"""Web product shell: live try-on stream + status API + garment swap.

IMPL: app/serve.py

This is the product form described in PRODUCT.md: a browser-first surface that
serves the running 30 fps loop as an MJPEG stream, exposes machine-readable
status, and lets a user try another garment without restarting.

Why a web shell first: it needs no app-store review, runs on the same machine
that owns the GPU, and can be embedded in a merchant's product page as an
<iframe>. A native app / mini-program would only re-wrap the same endpoints.

Run:
    python -m app.serve --port 8000
    python -m app.serve --physics-solver warp --render-backend opencv

Endpoints:
    GET  /                 -> single-page UI
    GET  /stream           -> multipart/x-mixed-replace MJPEG
    GET  /api/status       -> JSON status (framing / material / physics / timing)
    GET  /api/frame.jpg    -> one JPEG (screenshot / fallback)
    POST /api/garment      -> multipart image upload, rebuilds pipeline A
    GET  /api/doctor       -> capability report (same data as `cli doctor`)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tryon.logging import setup_logging  # noqa: E402

INDEX_HTML = """<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实时布料试穿 · Live Try-on</title>
<style>
  :root{--bg:#0f1216;--panel:#171c22;--line:#242c35;--fg:#e6edf5;--dim:#8b98a8;--accent:#4ea1ff;--ok:#3ddc97;--warn:#ffb454;--bad:#ff6b6b}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,"Segoe UI",Roboto,"PingFang SC",sans-serif}
  header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
  header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.02em}
  header .tag{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:999px;padding:2px 9px}
  main{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:16px;padding:16px;max-width:1280px;margin:0 auto}
  @media (max-width:880px){main{grid-template-columns:1fr}}
  .stage{position:relative;background:#000;border:1px solid var(--line);border-radius:12px;overflow:hidden;aspect-ratio:16/9}
  .stage img{width:100%;height:100%;object-fit:contain;display:block;background:#000}
  .badge{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap}
  .chip{font-size:11px;padding:3px 8px;border-radius:6px;background:rgba(10,14,18,.78);border:1px solid var(--line)}
  .chip.ok{color:var(--ok);border-color:rgba(61,220,151,.35)}
  .chip.warn{color:var(--warn);border-color:rgba(255,180,84,.35)}
  .chip.bad{color:var(--bad);border-color:rgba(255,107,107,.35)}
  aside{display:flex;flex-direction:column;gap:12px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}
  .card h2{font-size:12px;margin:0 0 10px;color:var(--dim);font-weight:600;letter-spacing:.08em;text-transform:uppercase}
  .row{display:flex;justify-content:space-between;gap:10px;padding:4px 0;font-size:13px}
  .row span:first-child{color:var(--dim)}
  .row span:last-child{font-variant-numeric:tabular-nums;text-align:right}
  .guide{font-size:12px;color:var(--warn);min-height:18px}
  .drop{border:1px dashed var(--line);border-radius:10px;padding:16px;text-align:center;color:var(--dim);cursor:pointer;transition:.15s}
  .drop:hover,.drop.over{border-color:var(--accent);color:var(--accent)}
  input[type=file]{display:none}
  .hint{font-size:11px;color:var(--dim);margin-top:8px;line-height:1.45}
  .spark{height:4px;background:var(--line);border-radius:99px;overflow:hidden;margin-top:6px}
  .spark>i{display:block;height:100%;background:var(--accent);width:0;transition:width .2s}
</style></head>
<body>
<header>
  <h1>实时布料试穿</h1>
  <span class="tag" id="engineTag">engine</span>
  <span class="tag">SPEC v2.0</span>
</header>
<main>
  <section class="stage">
    <img id="live" src="/stream" alt="live try-on stream">
    <div class="badge">
      <span class="chip" id="chipFraming">framing</span>
      <span class="chip" id="chipMaterial">material</span>
      <span class="chip" id="chipFps">fps</span>
    </div>
  </section>
  <aside>
    <div class="card">
      <h2>取景 / Framing</h2>
      <div class="row"><span>verdict</span><span id="vFraming">-</span></div>
      <div class="row"><span>可见关键点</span><span id="vVis">-</span></div>
      <div class="guide" id="vGuide"></div>
    </div>
    <div class="card">
      <h2>资产 / Asset</h2>
      <div class="row"><span>id</span><span id="vAsset">-</span></div>
      <div class="row"><span>version</span><span id="vVersion">-</span></div>
      <div class="row"><span>置信度</span><span id="vConf">-</span></div>
      <div class="row"><span>材质状态</span><span id="vMaterial">-</span></div>
      <div class="row"><span>附着/穿透</span><span id="vPhys">-</span></div>
    </div>
    <div class="card">
      <h2>换一件衣服</h2>
      <label class="drop" id="drop">
        点击或拖入衣服图片
        <input type="file" id="file" accept="image/*">
      </label>
      <div class="spark"><i id="bar"></i></div>
      <div class="hint" id="swapHint">单张正面平铺图效果最好；背面由先验补全，HUD 会如实标注。</div>
    </div>
    <div class="card">
      <h2>性能 / Timing</h2>
      <div class="row"><span>display fps</span><span id="vFps">-</span></div>
      <div class="row"><span>physics Hz</span><span id="vHz">-</span></div>
      <div class="row"><span>p95 延迟</span><span id="vP95">-</span></div>
    </div>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
function cls(el, ok){ el.className = 'chip ' + (ok ? 'ok' : 'warn'); }
async function tick(){
  try{
    const r = await fetch('/api/status'); const s = await r.json();
    if(!s.ready){ return; }
    $('vFraming').textContent = s.framing.verdict;
    $('vVis').textContent = (s.framing.visible_keypoint_ratio*100).toFixed(0) + '%';
    $('vGuide').textContent = s.framing.guide_text || '';
    $('chipFraming').textContent = 'framing: ' + s.framing.verdict;
    cls($('chipFraming'), s.framing.verdict === 'OK');
    $('chipMaterial').textContent = 'material: ' + s.material_state;
    cls($('chipMaterial'), s.material_state === 'CALIBRATED');
    const fps = s.timing.display_fps || 0;
    $('chipFps').textContent = fps.toFixed(1) + ' fps';
    cls($('chipFps'), fps >= 24);
    $('vAsset').textContent = s.asset.id;
    $('vVersion').textContent = 'v' + s.asset.version;
    $('vConf').textContent = (s.asset.confidence.overall*100).toFixed(0) + '%';
    $('vMaterial').textContent = s.material_state;
    $('vPhys').textContent = s.physics.attached + ' / ' + s.physics.penetration;
    $('vFps').textContent = fps.toFixed(1);
    $('vHz').textContent = (s.timing.physics_hz||0).toFixed(1);
    $('vP95').textContent = (s.timing.capture_to_display_ms_p95||0).toFixed(1) + ' ms';
  }catch(e){}
}
setInterval(tick, 500); tick();
const drop = $('drop'), file = $('file'), bar = $('bar'), hint = $('swapHint');
drop.addEventListener('dragover', e=>{e.preventDefault();drop.classList.add('over')});
drop.addEventListener('dragleave', ()=>drop.classList.remove('over'));
drop.addEventListener('drop', e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files.length) upload(e.dataTransfer.files[0]);});
file.addEventListener('change', e=>{ if(e.target.files.length) upload(e.target.files[0]); });
async function upload(f){
  bar.style.width='15%'; hint.textContent='管线 A 重建资产中…';
  const fd = new FormData(); fd.append('image', f);
  try{
    const r = await fetch('/api/garment', {method:'POST', body:fd});
    const j = await r.json();
    bar.style.width='100%';
    hint.textContent = j.ok
      ? `已切换：${j.asset.asset_id} · ${j.asset.mesh.n_vertices} 顶点 · G1 ${j.asset.g1_pass?'通过':'未通过'}`
      : ('失败：' + (j.error||'unknown'));
  }catch(e){ hint.textContent = '上传失败：' + e; }
  setTimeout(()=>{bar.style.width='0'}, 1200);
}
fetch('/api/config').then(r=>r.json()).then(c=>{
  $('engineTag').textContent = `${c.physics} · ${c.render} · ${c.geometry}`;
}).catch(()=>{});
</script>
</body></html>
"""


class LiveSession:
    """Single producer thread advancing the runner; readers get the last frame."""

    def __init__(self, cfg, garment_image: np.ndarray, target_fps: float = 30.0) -> None:
        from app.main import MockRunner

        self.cfg = cfg
        self.runner = MockRunner(
            cfg,
            garment_image=garment_image,
            log_path=Path(cfg.get("serve.log_path", REPO_ROOT / "var/logs/serve.jsonl")),
            headless=True,
            enable_keyframes=True,
            enable_calibration=True,
        )
        self.period = 1.0 / max(target_fps, 1.0)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tryon-live", daemon=True)
        self.frames = 0
        self._jpeg: bytes | None = None
        self._jpeg_id = -1

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        setup_logging()
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                with self._lock:
                    self.runner._step_frame()
                    self.frames += 1
            except Exception:  # noqa: BLE001 - keep the stream alive
                time.sleep(0.05)
            next_t += self.period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    # -- readers ---------------------------------------------------------
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            frame_id = getattr(self.runner, "frame_id", 0)
            image = getattr(self.runner.display, "last_image", None)
            if image is None:
                return None
            if frame_id == self._jpeg_id and self._jpeg is not None:
                return self._jpeg
            self._jpeg = _encode_jpeg(image)
            self._jpeg_id = frame_id
            return self._jpeg

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self.runner.status_payload()

    def swap(self, image: np.ndarray) -> dict[str, Any]:
        with self._lock:
            return self.runner.swap_garment(image)


def _encode_jpeg(image: np.ndarray) -> bytes:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        import cv2

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            return buf.tobytes()
    except Exception:
        pass
    from PIL import Image

    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="JPEG", quality=82)
    return out.getvalue()


def create_app(session: LiveSession):
    from flask import Flask, Response, jsonify, request, send_from_directory

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

    @app.get("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")

    @app.get("/api/config")
    def api_config():
        return jsonify(
            {
                "mode": session.cfg.get("mode"),
                "physics": session.cfg.get("physics.solver"),
                "render": session.cfg.get("render.backend"),
                "geometry": session.cfg.get("asset.geometry"),
                "keyframe": session.cfg.get("keyframe.backend"),
            }
        )

    @app.get("/api/status")
    def api_status():
        return jsonify(session.status())

    @app.get("/api/frame.jpg")
    def api_frame():
        blob = session.latest_jpeg()
        if blob is None:
            return Response(status=503)
        return Response(blob, mimetype="image/jpeg")

    @app.get("/stream")
    def stream():
        def gen():
            while True:
                blob = session.latest_jpeg()
                if blob is not None:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(
                        len(blob)
                    ).encode() + b"\r\n\r\n" + blob + b"\r\n"
                time.sleep(1.0 / 30.0)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/garment")
    def api_garment():
        from PIL import Image

        f = request.files.get("image")
        if f is None:
            return jsonify({"ok": False, "error": "no image part"}), 400
        try:
            img = np.asarray(Image.open(f.stream).convert("RGB"), dtype=np.uint8)
        except Exception as exc:  # noqa: BLE001 - user input boundary
            return jsonify({"ok": False, "error": f"decode failed: {exc}"}), 400
        if img.shape[0] < 32 or img.shape[1] < 32:
            return jsonify({"ok": False, "error": "image smaller than 32px"}), 400
        try:
            result = session.swap(img)
        except Exception as exc:  # noqa: BLE001 - pipeline A may reject the image
            code = getattr(exc, "code", type(exc).__name__)
            return jsonify({"ok": False, "error": str(exc), "code": code}), 422
        return jsonify({"ok": True, "asset": result})

    @app.get("/api/doctor")
    def api_doctor():
        from app.cli import cmd_doctor

        import contextlib

        buf = io.StringIO()
        ns = argparse.Namespace(
            mode=session.cfg.get("mode"),
            hardware=session.cfg.get("hardware"),
            config_dir=None,
        )
        with contextlib.redirect_stdout(buf):
            cmd_doctor(ns)
        try:
            payload = json.loads(buf.getvalue())
        except Exception:
            return jsonify({"raw": buf.getvalue()})
        # cmd_doctor reloads defaults; report what THIS server is actually using.
        payload["selected"] = {
            "mode": session.cfg.get("mode"),
            "physics": session.cfg.get("physics.solver"),
            "render": session.cfg.get("render.backend"),
            "geometry": session.cfg.get("asset.geometry"),
            "keyframe": session.cfg.get("keyframe.backend"),
            "grad": session.cfg.get("calibration.grad_backend"),
        }
        return jsonify(payload)

    @app.get("/assets/samples/<path:name>")
    def assets(name):
        return send_from_directory(REPO_ROOT / "assets" / "samples", name)

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web product shell for the try-on loop")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", default="mock", choices=["mock", "real", "hybrid"])
    p.add_argument("--hardware", default=None)
    p.add_argument("--garment", default=str(REPO_ROOT / "assets" / "samples" / "garment.png"))
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--physics-solver", default=None, choices=["stub_xpbd", "warp", "diffcloth"])
    p.add_argument("--render-backend", default=None, choices=["stub", "opencv", "3dgs", "gaussian"])
    p.add_argument("--geometry", default=None,
                   choices=["stub", "silhouette", "pattern", "sewformer", "instantmesh", "garmentrec"])
    p.add_argument("--mesh-resolution", nargs=2, type=int, default=None, metavar=("AROUND", "HEIGHT"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from app.main import load_garment_image
    from tryon.config import load_config

    cfg = load_config(args.mode, args.hardware)
    cfg.set("camera.width", int(args.width))
    cfg.set("camera.height", int(args.height))
    if args.physics_solver:
        cfg.set("physics.solver", args.physics_solver)
    if args.render_backend:
        cfg.set("render.backend", args.render_backend)
    if args.geometry:
        cfg.set("asset.geometry", args.geometry)
    if args.mesh_resolution:
        cfg.set("asset.mesh_resolution", [int(args.mesh_resolution[0]), int(args.mesh_resolution[1])])

    garment = load_garment_image(args.garment, cfg)
    session = LiveSession(cfg, garment, target_fps=float(args.fps))
    session.start()
    app = create_app(session)
    print(f"[serve] http://{args.host}:{args.port}  ({cfg.get('physics.solver')} · "
          f"{cfg.get('render.backend')} · {cfg.get('asset.geometry')})")
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        session.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
