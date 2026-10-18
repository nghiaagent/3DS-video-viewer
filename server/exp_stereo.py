"""Experiments on one clip:
  1. two eyes vs left only (flicker on static pixels)
  2. 3DS1 disparity test: does measured left/right disparity match the metadata, and can ONE global affine map
     DA3 inverse depth -> disparity across the whole clip (vs a per-frame fit)?

  server/.venv/bin/python server/exp_stereo.py "<path>.AVI"
"""
import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import serve
from bench_maxviews import decode_eye
from demo_clip import flicker, perframe_baseline

MIN_D, NUM_D = -96, 64  # searched disparity d in [-96, -32): R[y, x-d] ~ L[y, x], i.e. right feature sits to the RIGHT


def read_3ds1(video: Path):
    head = video.read_bytes()[: 256 * 1024]
    i = head.find(b"3DS1")
    if i < 0:
        return None
    off = struct.unpack("<f", head[i + 40: i + 44])[0]
    return off, round(off * 1.75)


def measure_disparity(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    """SGBM disparity in 480x240 pixels (negative values), NaN where invalid."""
    sg = cv2.StereoSGBM_create(minDisparity=MIN_D, numDisparities=NUM_D, blockSize=7, P1=8 * 49, P2=32 * 49,
                               disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    d = sg.compute(g(left_rgb), g(right_rgb)).astype(np.float32) / 16.0
    d[d <= MIN_D] = np.nan  # OpenCV marks invalid as minDisparity-1
    return d


def up(inv_frame: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(inv_frame)[None, None]
    return F.interpolate(t, size=(serve.OUT_H, serve.OUT_W), mode="bilinear", align_corners=False)[0, 0].numpy()


def clip_norm(inv: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(inv[:: max(1, len(inv) // 32)].ravel(), [2, 98])
    return np.clip((inv - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--step", type=int, default=4, help="measure disparity every Nth frame")
    a = ap.parse_args()

    meta = read_3ds1(a.video)
    print(f"3DS1 metadata: parallaxOffset={meta[0]:.2f} -> predicted base disparity {meta[1]} px" if meta else "no 3DS1 metadata")

    left, right = decode_eye(a.video, 0, serve.RES_W, serve.RES_H), decode_eye(a.video, 1, serve.RES_W, serve.RES_H)
    n = min(len(left), len(right)); left, right = left[:n], right[:n]
    luma = left.astype(np.float32).mean(-1)
    L480, R480 = decode_eye(a.video, 0, serve.OUT_W, serve.OUT_H), decode_eye(a.video, 1, serve.OUT_W, serve.OUT_H)

    variants = {}
    variants["stereo, 128 frames/pass"] = serve.infer_windows(left, right)
    print("stereo done", flush=True)
    variants["left-only, 128 frames/pass"] = serve.infer_windows(left, right, stereo=False, frames_per_pass=128)
    print("left-only (128) done", flush=True)
    variants[f"left-only, single pass ({n} views)"] = serve.infer_windows(left, right, stereo=False, frames_per_pass=n)
    print("left-only (single) done", flush=True)
    base_norm = perframe_baseline(left, right)
    print("baseline done", flush=True)

    # ---- 1. flicker ----
    print("\n== 1. two eyes vs left only: frame-to-frame depth change on static pixels (lower = steadier) ==")
    normed = {k: clip_norm(v) for k, v in variants.items()}
    normed["OLD per-frame pairs + per-frame stretch"] = base_norm
    for k, v in normed.items():
        f = flicker(v, luma)
        print(f"  {k:42s} static_MAD={f['static_MAD']:.4f}  p95={f['static_MAD_p95']:.4f}")

    # ---- 2. 3DS1 disparity ----
    idx = list(range(0, n, a.step))
    disp = {t: measure_disparity(L480[t], R480[t]) for t in idx}
    pooled = np.concatenate([d[np.isfinite(d)] for d in disp.values()])
    print("\n== 2. 3DS1 disparity test ==")
    print(f"  measured SGBM disparity over {len(idx)} frames: valid={np.mean([np.isfinite(d).mean() for d in disp.values()]):.0%} "
          f"median={np.median(pooled):.1f}  p2={np.percentile(pooled, 2):.1f}  p98={np.percentile(pooled, 98):.1f} px"
          + (f"   (metadata predicts base {meta[1]}, note says scene spans -74..-45)" if meta else ""))

    srcs = dict(variants); srcs["OLD per-frame pairs + per-frame stretch"] = base_norm
    print("  fit disparity ~ a*invdepth+b  (median |error| in px; per-frame fit vs ONE global fit for the whole clip)")
    for name, inv in srcs.items():
        per, glob_err, S, D = [], [], [], []
        for t in idx:
            d = disp[t]; ok = np.isfinite(d)
            if ok.mean() < 0.05:
                continue
            x = up(inv[t])[ok]; y = d[ok]
            S.append(x); D.append(y)
        Sall, Dall = np.concatenate(S), np.concatenate(D)
        ga, gb = serve.fit_affine(Sall, Dall)
        for x, y in zip(S, D):
            pa, pb = serve.fit_affine(x, y)
            per.append(np.median(np.abs(pa * x + pb - y)))
            glob_err.append(np.median(np.abs(ga * x + gb - y)))
        lo, hi = np.percentile(Sall, [2, 98])
        print(f"  {name:42s} per-frame={np.mean(per):5.2f}px  global={np.mean(glob_err):5.2f}px  "
              f"(global/per-frame={np.mean(glob_err) / max(np.mean(per), 1e-6):.2f})  implied range {ga * lo + gb:.0f}..{ga * hi + gb:.0f}px")


if __name__ == "__main__":
    main()
