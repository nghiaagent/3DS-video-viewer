"""Long-range consistency: depth change between frames k apart (1-3 s) on pixels that stayed static in between.

  server/.venv/bin/python server/exp_drift.py "<a>.AVI" "<b>.AVI" [--passes 32,64,128] [--lags 20,40,60]

Pairs (t, t+k) are split into "straddles a window boundary" vs "inside one window" so cross-seam drift shows up.
A pixel counts as static only if its luma stays within --dev of frame t for the whole interval.
"""
import argparse
from pathlib import Path

import numpy as np

import serve
from bench_maxviews import decode_eye
from demo_clip import perframe_baseline
from exp_stereo import clip_norm


def seam_starts(frames: int, fpp: int):
    overlap = max(2, fpp // 4)
    return [s for s in range(0, max(1, frames - overlap), fpp - overlap)][1:] if frames > fpp else []


def drift(norm, luma, lag, seams, dev, stride=5, min_static=0.02):
    inside, across = [], []
    for t in range(0, len(norm) - lag, stride):
        d = np.abs(luma[t + 1: t + lag + 1] - luma[t]).max(axis=0)
        static = d < dev
        if static.mean() < min_static:
            continue
        err = float(np.abs(norm[t + lag] - norm[t])[static].mean())
        (across if any(t < s <= t + lag for s in seams) else inside).append(err)
    return inside, across


def fmt(xs):
    return f"{np.mean(xs):.4f} (n={len(xs)})" if xs else "   n/a (n=0)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--passes", default="32,64,128")
    ap.add_argument("--lags", default="20,40,60")
    ap.add_argument("--dev", type=float, default=8.0)
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    lags = [int(x) for x in a.lags.split(",")]
    for v in a.videos:
        left, right = decode_eye(v, 0, serve.RES_W, serve.RES_H), decode_eye(v, 1, serve.RES_W, serve.RES_H)
        n = min(len(left), len(right)); left, right = left[:n], right[:n]
        luma = left.astype(np.float32).mean(-1)
        print(f"\n== {v.name}: {n} frames; mean |depth change| on static pixels, inside one window / across a seam ==", flush=True)
        print("  variant".ljust(40) + "".join(f"lag {k / 20:.0f}s".ljust(44) for k in lags))
        rows = {}
        for fpp in [int(x) for x in a.passes.split(",")]:
            try:
                inv = serve.infer_windows(left, right, frames_per_pass=fpp)
            except RuntimeError as e:
                print(f"  stereo {fpp} frames/pass: OOM at the memory cap ({str(e)[:60]})", flush=True)
                continue
            rows[f"stereo {fpp} frames/pass"] = (clip_norm(inv), seam_starts(n, fpp))
        if not a.skip_baseline:
            rows["OLD per-frame + per-frame stretch"] = (perframe_baseline(left, right), [])
        for name, (norm, seams) in rows.items():
            cells = []
            for k in lags:
                ins, acr = drift(norm, luma, k, seams, a.dev)
                cells.append(f"{fmt(ins)} / {fmt(acr)}".ljust(44))
            print(f"  {name:38s}" + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
