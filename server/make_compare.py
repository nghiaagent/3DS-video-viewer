"""Side-by-side comparison videos: picture | one depth panel per variant.

  DA3_RES_W=252 DA3_RES_H=126 server/.venv/bin/python server/make_compare.py "<a>.AVI" "<b>.AVI" \
      --variants stereo:64,stereo:128,mono:256 --out server/cache/compare

A variant is mode:frames_per_pass, mode = stereo (both eyes as views) or mono (left eye only).
Variants that run out of memory under the GPU cap (DA3_MAX_GB, default 7) are reported and skipped.
Depth uses the same clip-global stretch as the server and DA3's Spectral colours (red = near, blue = far).
Printed per variant: static-pixel flicker (frame-to-frame change, lower = steadier) AND depth contrast (mean spatial std
of the normalised depth). Flicker alone rewards flat depth, so read the two together; the final judge is the video.
"""
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

import serve
from bench_maxviews import FFMPEG, decode_eye
from demo_clip import flicker, spectral_lut, upscale_u8
from exp_stereo import clip_norm


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = np.array(img, copy=True)  # frombuffer frames are read-only; OpenCV needs a writable copy
    cv2.rectangle(out, (0, 0), (len(text) * 9 + 12, 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_compare(path: Path, video: Path, depths: list[np.ndarray], names: list[str], fps: float):
    w, h = serve.OUT_W, serve.OUT_H
    lut = spectral_lut()
    dec = subprocess.Popen([FFMPEG, "-v", "error", "-i", str(video), "-map", "0:v:0", "-vf", f"scale={w}:{h}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w * (1 + len(depths))}x{h}", "-r", str(fps), "-i", "-",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)], stdin=subprocess.PIPE)
    for i in range(len(depths[0])):
        buf = dec.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        panels = [label(rgb, "video (left eye)")] + [label(lut[d[i]], n) for d, n in zip(depths, names)]
        enc.stdin.write(np.concatenate(panels, axis=1).tobytes())
    enc.stdin.close(); enc.wait(); dec.stdout.close(); dec.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path, nargs="+")
    ap.add_argument("--variants", default="stereo:16,stereo:64")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "cache/compare")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    variants = [(v.split(":")[0], int(v.split(":")[1])) for v in args.variants.split(",")]
    print(f"resolution {serve.RES_W}x{serve.RES_H}, variants {variants}", flush=True)
    for v in args.videos:
        left, right = decode_eye(v, 0, serve.RES_W, serve.RES_H), decode_eye(v, 1, serve.RES_W, serve.RES_H)
        n = min(len(left), len(right)); left, right = left[:n], right[:n]
        luma = left.astype(np.float32).mean(-1)
        panels, names = [], []
        for mode, fpp in variants:
            tag = f"{mode} {fpp} frames/pass"
            try:
                inv = serve.infer_windows(left, right, stereo=(mode == "stereo"), frames_per_pass=fpp)
            except RuntimeError as e:
                print(f"{v.name}: {tag:26s} OUT OF MEMORY under the {serve.MAX_GB:g}GB cap ({str(e)[:50]}...)", flush=True)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                continue
            norm = clip_norm(inv)
            f = flicker(norm, luma)
            contrast = float(np.mean([norm[t].std() for t in range(0, n, 5)]))
            print(f"{v.name}: {tag:26s} flicker={f['static_MAD']:.4f}  contrast={contrast:.3f}  "
                  f"flicker/contrast={f['static_MAD'] / contrast:.3f}  (gpu peak {serve.PEAK_GB:.1f}GB)", flush=True)
            panels.append(upscale_u8(norm)); names.append(tag)
        if panels:
            out = args.out / f"{v.stem}_{'_'.join(f'{m[0]}{k}' for m, k in variants)}.mp4"
            write_compare(out, v, panels, names, serve.video_fps(v))
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
