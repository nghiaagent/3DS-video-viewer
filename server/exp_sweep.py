"""Window-size sweep: static-pixel flicker vs frames per pass (stereo), per clip, with the old behaviour as reference.

  server/.venv/bin/python server/exp_sweep.py "<a>.AVI" "<b>.AVI" [--passes 32,64,128]
"""
import argparse
import time
from pathlib import Path

import numpy as np

import serve
from bench_maxviews import decode_eye
from demo_clip import flicker, perframe_baseline
from exp_stereo import clip_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--passes", default="32,64,128")
    ap.add_argument("--skip-baseline", action="store_true")
    a = ap.parse_args()
    for v in a.videos:
        left, right = decode_eye(v, 0, serve.RES_W, serve.RES_H), decode_eye(v, 1, serve.RES_W, serve.RES_H)
        n = min(len(left), len(right)); left, right = left[:n], right[:n]
        luma = left.astype(np.float32).mean(-1)
        print(f"\n== {v.name}: {n} frames ==", flush=True)
        for fpp in [int(x) for x in a.passes.split(",")]:
            t = time.time()
            try:
                inv = serve.infer_windows(left, right, frames_per_pass=fpp)
            except RuntimeError as e:
                print(f"  stereo {fpp:3d} frames/pass: OOM at the memory cap ({str(e)[:60]})", flush=True)
                continue
            f = flicker(clip_norm(inv), luma)
            print(f"  stereo {fpp:3d} frames/pass: static_MAD={f['static_MAD']:.4f} p95={f['static_MAD_p95']:.4f} "
                  f"({time.time() - t:.0f}s, gpu peak {serve.PEAK_GB:.1f}GB)", flush=True)
        if a.skip_baseline:
            continue
        t = time.time()
        f = flicker(perframe_baseline(left, right), luma)
        print(f"  OLD per-frame + per-frame stretch: static_MAD={f['static_MAD']:.4f} p95={f['static_MAD_p95']:.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
