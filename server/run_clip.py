"""Run the server's windowed DA3 pipeline on one clip and report time, peak memory and seam continuity.

  server/.venv/bin/python server/run_clip.py "<path to .AVI>" [--max-frames N]

Seam metric: mean |depth[t]-depth[t-1]| on the final uint8 depth, compared at window seams vs everywhere else.
A good alignment gives seam/other ~ 1.0; a scale jump at the seams shows up as a ratio well above 1.
"""
import argparse
import resource
import time
from pathlib import Path

import numpy as np
import torch

import serve  # noqa: E402  (applies the MPS memory cap on first get_net())


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--dump", default="", help="PNG path: left frame + depth for the biggest step (k-1..k+2)")
    ap.add_argument("--max-frames", type=int, default=0, help="only use the first N frames")
    a = ap.parse_args()

    t0 = time.time()
    left = serve.decode_eye(a.video, 0, serve.RES_W, serve.RES_H)
    right = serve.decode_eye(a.video, 1, serve.RES_W, serve.RES_H)
    if a.max_frames:
        left, right = left[: a.max_frames], right[: a.max_frames]
    frames = min(len(left), len(right))
    print(f"{a.video.name}: {frames} frames decoded in {time.time() - t0:.0f}s, rss={rss_gb():.2f}GB", flush=True)

    t1 = time.time()
    inv = serve.infer_windows(left, right)
    infer_s = time.time() - t1
    peak_mps = serve.PEAK_GB
    depth = serve.quantize_and_upscale(inv)  # uint8; metrics below are chunked to keep memory low

    per_win = max(2, serve.MAX_VIEWS // 2)
    overlap = max(2, per_win // 4)
    seams = [s for s in range(per_win - overlap, frames, per_win - overlap)]  # start of each later window
    diff = np.array([np.abs(depth[i + 1].astype(np.int16) - depth[i]).mean() for i in range(frames - 1)])  # diff[i] = frame i+1 vs i
    seam_idx = sorted({min(len(diff) - 1, max(0, k + j)) for k in seams for j in range(0, overlap)})
    mask = np.zeros(len(diff), bool)
    mask[seam_idx] = True
    seam_mean = diff[mask].mean() if mask.any() else float("nan")
    other_mean = diff[~mask].mean()
    print(f"windows={len(seams) + (1 if frames > per_win else 0)} views/pass<={serve.MAX_VIEWS} infer={infer_s:.0f}s ({infer_s / frames * 1000:.0f} ms/frame) "
          f"peak_mps={peak_mps:.1f}GB peak_rss={rss_gb():.2f}GB")
    k = int(diff.argmax())
    luma = np.abs(left[k + 1].astype(np.float32).mean(-1) - left[k].astype(np.float32).mean(-1)).mean()
    print(f"biggest step at frame {k}->{k + 1}: depth {diff[k]:.1f}, left-eye luma diff {luma:.1f}, "
          f"in seam={bool(mask[k])}; top5 steps: {[(int(i), round(float(diff[i]), 1)) for i in np.argsort(diff)[-5:][::-1]]}")
    if a.dump:
        from PIL import Image
        cols = []
        for f in range(max(0, k - 1), min(frames, k + 3)):
            rgb = np.array(Image.fromarray(left[f]).resize((serve.OUT_W, serve.OUT_H)))
            dep = np.repeat(depth[f].astype(np.uint8)[..., None], 3, axis=-1)
            cols.append(np.concatenate([rgb, dep], axis=0))
        Image.fromarray(np.concatenate(cols, axis=1)).save(a.dump)
        print("saved", a.dump)
    print(f"total peak usage ~ {peak_mps + rss_gb():.1f}GB (mps {peak_mps:.1f} + rss {rss_gb():.1f}); target <= 8GB")
    print(f"frame-to-frame MAD: seams={seam_mean:.3f} other={other_mean:.3f} ratio={seam_mean / other_mean:.2f} "
          f"(max single step {diff.max():.2f})")


if __name__ == "__main__":
    main()
