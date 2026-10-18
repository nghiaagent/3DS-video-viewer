"""How many views (both eyes of N/2 frames) can DA3 take in ONE forward pass on this machine?

Usage (from repo root):
  server/.venv/bin/python server/bench_maxviews.py --res 252x126 --dtype bf16 --views 32,64,128,256,438

Each run is validated: output finite and non-constant. With --cpu-check up to N views, MPS output is also compared to a
CPU fp32 run at the same N (the real faithfulness test; correlation vs an N=2 run is NOT used because a view's depth
legitimately depends on which other views are in the pass).
Stops at the first failure for that (res, dtype).
"""
import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from da3_loader import load_model

FFMPEG = "/opt/homebrew/bin/ffmpeg"
DEFAULT_VIDEO = Path.home() / "Downloads/3DS vids/DCIM/101NIN03/HNI_0062.AVI"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def decode_eye(video: Path, stream: int, w: int, h: int) -> np.ndarray:
    """All frames of one eye as uint8 [F, h, w, 3]. The AVI holds the eyes as two separate video streams."""
    cmd = [FFMPEG, "-v", "error", "-i", str(video), "-map", f"0:v:{stream}",
           "-vf", f"scale={w}:{h}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def build_views(left: np.ndarray, right: np.ndarray, n_views: int) -> torch.Tensor:
    """Interleaved L0,R0,L1,R1,... normalized -> [1, N, 3, h, w]."""
    frames = min(len(left), len(right))
    idx = [(v >> 1) % frames for v in range(n_views)]
    src = [left if v % 2 == 0 else right for v in range(n_views)]
    arr = np.stack([src[v][idx[v]] for v in range(n_views)]).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).contiguous()


def build_mono_views(frames: np.ndarray) -> torch.Tensor:
    """Left-eye-only views -> [1, N, 3, h, w]."""
    arr = (frames.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).contiguous()


def run_model(net, x: torch.Tensor, dtype: str):
    ctx = (torch.autocast(device_type=x.device.type, dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[dtype])
           if dtype != "fp32" else torch.autocast(device_type=x.device.type, enabled=False))
    with torch.inference_mode(), ctx:
        return net(x, None, None, [], False, False, "saddle_balanced")


def depth_of(out) -> torch.Tensor:
    d = out["depth"] if "depth" in out else out["predicted_depth"]
    return d.float().squeeze(0).squeeze(-1) if d.ndim == 5 else d.float().squeeze(0)  # [N, h, w]


def sync(dev: str):
    if dev == "mps":
        torch.mps.synchronize()


def mps_mem_gb() -> float:
    return torch.mps.driver_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    ap.add_argument("--res", default="252x126")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--views", default="32,64,128,256,438")
    ap.add_argument("--cpu-check", type=int, default=0, help="also run CPU fp32 at the same N for N <= this and report corr")
    ap.add_argument("--max-gb", type=float, default=7.0, help="hard cap on MPS memory; larger runs are skipped/OOM instead of straining the machine")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    w, h = map(int, a.res.split("x"))
    assert w % 14 == 0 and h % 14 == 0, "ViT-14 needs multiples of 14"

    if a.device == "mps":
        total_gb = torch.mps.recommended_max_memory() / 1e9
        torch.mps.set_per_process_memory_fraction(min(1.0, a.max_gb / total_gb))
    model, _ = load_model(a.model, a.device)
    net = model.model
    left, right = decode_eye(a.video, 0, w, h), decode_eye(a.video, 1, w, h)
    print(f"device={a.device} res={w}x{h} dtype={a.dtype} frames={len(left)}/{len(right)} (eyes differ: {not np.array_equal(left[0], right[0])})")

    cpu_net = load_model(a.model, "cpu")[0].model if a.cpu_check else None
    last_n, last_gb = 0, 0.0
    for n in [int(v) for v in a.views.split(",")]:
        if last_n and last_gb * n / last_n > a.max_gb:
            print(f"N={n:4d} SKIPPED projected {last_gb * n / last_n:.1f}GB > --max-gb {a.max_gb}", flush=True)
            break
        try:
            x = build_views(left, right, n).to(a.device)
            if a.device == "mps":
                torch.mps.empty_cache()
            sync(a.device)
            t = time.time()
            d = depth_of(run_model(net, x, a.dtype))
            sync(a.device)
            dt = time.time() - t
            dc = d.cpu()
            finite = bool(torch.isfinite(dc).all())
            spread = float(dc.std())
            ok = finite and spread > 1e-4
            cpu_txt = ""
            if cpu_net is not None and n <= a.cpu_check:
                t0 = time.time()
                ref = depth_of(run_model(cpu_net, build_views(left, right, n), "fp32"))
                c = [float(np.corrcoef(dc[i].flatten().numpy(), ref[i].flatten().numpy())[0, 1]) for i in range(n)]
                cpu_txt = f" cpu_corr min={min(c):.4f} mean={np.mean(c):.4f} ({time.time() - t0:.0f}s cpu)"
                ok = ok and min(c) > 0.95
            print(f"N={n:4d} {'VALID' if ok else 'INVALID'} {dt:7.1f}s mem={mps_mem_gb():.1f}GB std={spread:.3f}{cpu_txt} out={tuple(dc.shape)}", flush=True)
            last_n, last_gb = n, mps_mem_gb()
            if not ok:
                break
        except Exception as e:  # OOM / unsupported op
            print(f"N={n:4d} FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
            break
        finally:
            x = d = dc = None


if __name__ == "__main__":
    main()
