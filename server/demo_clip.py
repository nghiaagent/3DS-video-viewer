"""One-off demo + flicker comparison for a clip.

  server/.venv/bin/python server/demo_clip.py "<path>.AVI" --out server/cache/demo

Outputs in --out:
  <name>_side_by_side.mp4   left = video (left eye), right = joint-pass depth (clip-global normalization, DA3 'Spectral' colours: red = near)
  <name>_perframe_depth.mp4 same layout but with the OLD behaviour (each frame inferred alone, per-frame p2/p98)
  <name>.glb                one-off DA3 point-cloud/camera export from evenly spaced frames
Prints flicker metrics for both depth sets on static pixels.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import serve
from bench_maxviews import FFMPEG, build_views, decode_eye, depth_of, run_model


def upscale_u8(norm: np.ndarray) -> np.ndarray:
    out = np.empty((len(norm), serve.OUT_H, serve.OUT_W), np.uint8)
    for i in range(0, len(norm), 32):
        t = torch.from_numpy(norm[i:i + 32]).unsqueeze(1)
        up = F.interpolate(t, size=(serve.OUT_H, serve.OUT_W), mode="bilinear", align_corners=False)
        out[i:i + 32] = (up.squeeze(1).numpy() * 255 + 0.5).astype(np.uint8)
    return out


def perframe_baseline(left, right):
    """Old behaviour: every frame is its own [L,R] pass and is normalised by its own p2/p98."""
    net, device = serve.get_net()
    out = np.empty((len(left), serve.RES_H, serve.RES_W), np.float32)
    for f in range(len(left)):
        x = build_views(left[f:f + 1], right[f:f + 1], 2).to(device)
        z = depth_of(run_model(net, x, serve.DTYPE))[0].cpu().numpy()
        inv = 1.0 / np.maximum(z, 1e-3)
        lo, hi = np.percentile(inv, [2, 98])
        out[f] = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def flicker(norm: np.ndarray, luma: np.ndarray, thresh: float = 4.0):
    """Frame-to-frame |depth change| on pixels whose left-eye luma barely changed (static background)."""
    mads, means = [], norm.reshape(len(norm), -1).mean(axis=1)
    for t in range(1, len(norm)):
        static = np.abs(luma[t] - luma[t - 1]) < thresh
        if static.mean() > 0.05:
            mads.append(float(np.abs(norm[t] - norm[t - 1])[static].mean()))
    return {"static_MAD": float(np.mean(mads)), "static_MAD_p95": float(np.percentile(mads, 95)),
            "pumping_std_of_frame_mean": float(means.std()), "frames_used": len(mads)}


def spectral_lut() -> np.ndarray:
    """Same colouring as DA3's depth_vis (utils/visualize.py): Spectral(1 - normalised inverse depth), near = red."""
    import matplotlib
    cm = matplotlib.colormaps["Spectral"]
    return (cm(1.0 - np.arange(256) / 255.0)[:, :3] * 255).astype(np.uint8)


def write_video(path: Path, video: Path, depth_u8: np.ndarray, fps: float):
    w, h = serve.OUT_W, serve.OUT_H
    lut = spectral_lut()
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf", f"scale={w}:{h}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)],
                           stdin=subprocess.PIPE)
    for i in range(len(depth_u8)):
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        dep = lut[depth_u8[i]]
        enc.stdin.write(np.concatenate([rgb, dep], axis=1).tobytes())
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()


def export_glb(video: Path, out_dir: Path, name: str, n_frames: int = 16):
    from PIL import Image
    from da3_loader import load_model
    frames_dir = out_dir / f"{name}_glb_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    left = decode_eye(video, 0, serve.OUT_W, serve.OUT_H)
    idx = np.linspace(0, len(left) - 1, n_frames).astype(int)
    paths = []
    for k, i in enumerate(idx):
        p = frames_dir / f"{k:03d}.png"
        Image.fromarray(left[i]).save(p)
        paths.append(str(p))
    # the shared net is on MPS already; the high-level API wants the full DepthAnything3 wrapper
    model, device = load_model()
    model.inference(paths, export_dir=str(out_dir / f"{name}_glb"), export_format="glb", process_res=504,
                    process_res_method="upper_bound_resize")
    glbs = list((out_dir / f"{name}_glb").rglob("*.glb"))
    if glbs:
        glbs[0].replace(out_dir / f"{name}.glb")
        return out_dir / f"{name}.glb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=Path("server/cache/demo"))
    ap.add_argument("--skip-glb", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    name = a.video.stem

    left = decode_eye(a.video, 0, serve.RES_W, serve.RES_H)
    right = decode_eye(a.video, 1, serve.RES_W, serve.RES_H)
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    luma = left.astype(np.float32).mean(-1)
    fps = serve.video_fps(a.video)

    inv = serve.infer_windows(left, right)
    lo, hi = np.percentile(inv[:: max(1, n // 32)].ravel(), [2, 98])
    joint = np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)
    print(f"joint pass done ({n} frames), gpu peak {serve.PEAK_GB:.1f}GB", flush=True)
    base = perframe_baseline(left, right)
    print("per-frame baseline done", flush=True)

    print("JOINT      ", {k: round(v, 4) for k, v in flicker(joint, luma).items()})
    print("PER-FRAME  ", {k: round(v, 4) for k, v in flicker(base, luma).items()})

    write_video(a.out / f"{name}_side_by_side.mp4", a.video, upscale_u8(joint), fps)
    write_video(a.out / f"{name}_perframe_depth.mp4", a.video, upscale_u8(base), fps)
    print("videos written to", a.out, flush=True)
    if not a.skip_glb:
        try:
            print("glb:", export_glb(a.video, a.out, name))
        except Exception as e:  # report, do not hide
            print(f"GLB export FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
