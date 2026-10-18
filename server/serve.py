"""Local DA3 depth server + static host for the built viewer.

  npm run build && server/.venv/bin/python server/serve.py
  open http://127.0.0.1:8000/?dcim=101NIN03/HNI_0062.AVI

All frames of the clip, both eyes, go through DA3-BASE in as few forward passes as the memory cap allows
(one pass when frames*2 <= MAX_VIEWS). Longer clips use overlapping windows chained by a robust affine fit in
inverse depth (same idea as src/depth/windowAlignment.ts). One clip-global [p2,p98] is applied at the end,
so there is no per-frame normalization.
"""
import hashlib
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from bench_maxviews import FFMPEG, build_mono_views, build_views, decode_eye, depth_of, run_model
from da3_loader import load_model

ROOT = Path(__file__).resolve().parent.parent
DCIM_ROOT = Path(os.environ.get("DCIM_ROOT", Path.home() / "Downloads/3DS vids/DCIM")).resolve()
CACHE = ROOT / "server/cache"
RES_W, RES_H = int(os.environ.get("DA3_RES_W", 252)), int(os.environ.get("DA3_RES_H", 126))  # 2:1, multiples of 14
OUT_W, OUT_H = 480, 240
MAX_GB = float(os.environ.get("DA3_MAX_GB", 7.0))  # GPU allocator cap; + ~1GB process memory stays under 8GB total
MAX_VIEWS = int(os.environ.get("DA3_MAX_VIEWS", 256))  # 128 frames x 2 eyes per pass (chosen by eye over 64); ~4.8GB at 252x126
DTYPE = os.environ.get("DA3_DTYPE", "bf16")

app = FastAPI()
_model = {}
PROGRESS = {"phase": "idle", "done": 0, "total": 0}  # polled by the viewer via /api/progress
PEAK_GB = 0.0  # max MPS driver allocation seen right after a pass (before empty_cache)
_lock = threading.Lock()  # one inference at a time; the GPU memory cap assumes it


@app.middleware("http")
async def isolation_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    return resp


def get_net():
    if "net" not in _model:
        if torch.backends.mps.is_available():
            total = torch.mps.recommended_max_memory() / 1e9
            torch.mps.set_per_process_memory_fraction(min(1.0, MAX_GB / total))
        model, device = load_model()
        _model.update(net=model.model, device=device)
    return _model["net"], _model["device"]


def safe_dcim(rel: str) -> Path:
    p = (DCIM_ROOT / os.path.normpath(rel)).resolve()
    if DCIM_ROOT not in p.parents or not p.is_file():
        raise HTTPException(404, "File not found or forbidden")
    return p


def video_fps(video: Path) -> float:
    out = subprocess.run([FFMPEG.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(video)],
                         capture_output=True, text=True, check=True).stdout.strip()
    num, den = out.split("/")
    return float(num) / float(den)


# ---- window alignment (numpy port of the affine chaining in src/depth/windowAlignment.ts) ----
def fit_affine(src: np.ndarray, dst: np.ndarray, iters: int = 5) -> tuple[float, float]:
    """Huber-IRLS fit dst ~ a*src + b on a subsample."""
    s, d = src.ravel(), dst.ravel()
    if s.size > 200_000:
        idx = np.random.default_rng(0).choice(s.size, 200_000, replace=False)
        s, d = s[idx], d[idx]
    ok = np.isfinite(s) & np.isfinite(d)
    s, d = s[ok], d[ok]
    w = np.ones_like(s)
    a, b = 1.0, 0.0
    for _ in range(iters):
        sw = w.sum()
        mx, my = (w * s).sum() / sw, (w * d).sum() / sw
        var = (w * (s - mx) ** 2).sum()
        a = (w * (s - mx) * (d - my)).sum() / var if var > 1e-12 else 1.0
        b = my - a * mx
        r = d - (a * s + b)
        k = 1.345 * max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        w = np.where(np.abs(r) <= k, 1.0, k / np.abs(r))
    return float(a), float(b)


def infer_windows(left: np.ndarray, right: np.ndarray, stereo: bool = True, frames_per_pass: int = 0) -> np.ndarray:
    """Returns aligned inverse depth [F, h, w] float32 for the left eye of every frame."""
    net, device = get_net()
    frames = min(len(left), len(right))
    per_win = frames_per_pass or max(2, MAX_VIEWS // 2 if stereo else MAX_VIEWS)
    overlap = max(2, per_win // 4)
    starts = [0] if frames <= per_win else list(range(0, max(1, frames - overlap), per_win - overlap))
    inv = np.zeros((frames, RES_H, RES_W), np.float32)
    prev_end = 0
    PROGRESS.update(phase="inferring", done=0, total=len(starts))
    for start in starts:
        end = min(frames, start + per_win)
        if prev_end and end <= prev_end:
            break
        if stereo:
            x = build_views(left[start:end], right[start:end], 2 * (end - start)).to(device)
            z = depth_of(run_model(net, x, DTYPE))[0::2].cpu().numpy()  # left-eye views only
        else:
            x = build_mono_views(left[start:end]).to(device)
            z = depth_of(run_model(net, x, DTYPE)).cpu().numpy()
        win = 1.0 / np.maximum(z, 1e-3)
        global PEAK_GB
        if device == "mps":
            PEAK_GB = max(PEAK_GB, torch.mps.driver_allocated_memory() / 1e9)
        del x
        if device == "mps":
            torch.mps.empty_cache()
        n_ov = max(0, prev_end - start)
        if n_ov:  # map this window into the already-aligned space using the shared frames
            a, b = fit_affine(win[:n_ov], inv[start:prev_end])
            win = a * win + b
            t = (np.arange(n_ov, dtype=np.float32) + 1) / (n_ov + 1)  # linear cross-fade over the overlap
            inv[start:prev_end] = (1 - t)[:, None, None] * inv[start:prev_end] + t[:, None, None] * win[:n_ov]
        inv[start + n_ov:end] = win[n_ov:]
        prev_end = end
        PROGRESS["done"] += 1
        if end >= frames:
            break
    return inv


def quantize_and_upscale(inv: np.ndarray) -> np.ndarray:
    """One clip-global [p2,p98] -> uint8 [F, OUT_H, OUT_W], larger = nearer."""
    samp = inv[:: max(1, len(inv) // 32)].ravel()
    lo, hi = np.percentile(samp, [2, 98])
    norm = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.empty((len(inv), OUT_H, OUT_W), np.uint8)
    for i in range(0, len(inv), 32):
        t = torch.from_numpy(norm[i:i + 32]).unsqueeze(1)
        up = F.interpolate(t, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
        out[i:i + 32] = (up.squeeze(1).numpy() * 255 + 0.5).astype(np.uint8)
    return out


def compute_depth(video: Path) -> bytes:
    PROGRESS.update(phase="decoding", done=0, total=0)
    left, right = decode_eye(video, 0, RES_W, RES_H), decode_eye(video, 1, RES_W, RES_H)
    inv = infer_windows(left, right)
    depth = quantize_and_upscale(inv)
    PROGRESS.update(phase="finishing")
    header = json.dumps({"frames": len(depth), "width": OUT_W, "height": OUT_H, "fps": video_fps(video),
                         "res": f"{RES_W}x{RES_H}", "maxViews": MAX_VIEWS}).encode()
    return struct.pack("<I", len(header)) + header + depth.tobytes()


# ---- routes ----
@app.get("/api/health")
def health():
    return {"model": "DA3-BASE", "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "maxViews": MAX_VIEWS, "res": f"{RES_W}x{RES_H}", "dtype": DTYPE}


@app.get("/api/progress")
def progress():
    return PROGRESS


@app.get("/api/dcim/list")
def dcim_list():
    items = []
    for p in sorted(DCIM_ROOT.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and p.suffix.upper() in (".AVI", ".MPO"):
            rel = p.relative_to(DCIM_ROOT)
            st = p.stat()
            items.append({"name": p.name, "relativePath": str(rel),
                          "folder": "" if str(rel.parent) == "." else str(rel.parent),
                          "sizeBytes": st.st_size, "mtime": st.st_mtime * 1000})
    return JSONResponse(items)


@app.get("/api/dcim/file")
def dcim_file(path: str):
    p = safe_dcim(path)
    mime = {".AVI": "video/x-msvideo", ".MPO": "image/mpo"}.get(p.suffix.upper(), "application/octet-stream")
    return FileResponse(p, media_type=mime)  # Starlette serves Range requests


def depth_bytes(video: Path) -> bytes:
    """Depth sequence for a video file (cached on disk by path + mtime + settings)."""
    key = hashlib.sha1(f"{video}|{video.stat().st_mtime_ns}|{RES_W}x{RES_H}|{MAX_VIEWS}|{DTYPE}".encode()).hexdigest()
    cached = CACHE / f"{key}.bin"
    if cached.exists():
        return cached.read_bytes()
    with _lock:
        if not cached.exists():
            try:
                blob = compute_depth(video)
            finally:
                PROGRESS.update(phase="idle", done=0, total=0)
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(blob)
    return cached.read_bytes()


@app.get("/api/depth")
def depth(path: str):
    return Response(depth_bytes(safe_dcim(path)), media_type="application/octet-stream")


@app.post("/api/depth/upload")
async def depth_upload(request: Request):
    """Depth for a file the user picked in the browser: the body is streamed to disk, then processed like a DCIM clip.

    Uploads are stored content-addressed in server/cache/uploads so the same file is only processed once.
    """
    uploads = CACHE / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    tmp = uploads / f"upload-{os.getpid()}-{time.time_ns()}.part"
    sha, size = hashlib.sha1(), 0
    with tmp.open("wb") as f:
        async for chunk in request.stream():
            sha.update(chunk)
            size += len(chunk)
            f.write(chunk)
    if size == 0:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")
    final = uploads / f"{sha.hexdigest()}.avi"
    if final.exists():
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(final)
    try:
        blob = await run_in_threadpool(depth_bytes, final)
    except Exception as e:  # not a 3DS dual-stream AVI, decode failure, ...
        raise HTTPException(422, f"could not compute depth: {type(e).__name__}: {str(e)[:200]}")
    return Response(blob, media_type="application/octet-stream")


dist = ROOT / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="viewer")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Viewer: http://127.0.0.1:{port}/?dcim=101NIN03/HNI_0062.AVI")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
