# 3DS Video Viewer

Browser viewer and exporter that turns Nintendo 3DS `.AVI` stereo recordings into a 2D parallax ("spatial 3D") video.
Depth comes from [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3), run **on your own
machine** by a small local server that uses the Apple GPU. The server processes every frame of the clip with both
eyes in windowed joint passes and one clip-wide depth scale, which is what keeps the depth steady over time.

## Quick start (macOS, Apple Silicon)

You need [Homebrew](https://brew.sh) tools once: `brew install node uv ffmpeg`.

```sh
npm install              # viewer dependencies
npm run server:setup     # one-time: Python 3.12 env + PyTorch + DA3 + model weights (~1.1 GB env, ~0.5 GB weights)
npm run serve            # builds the viewer and starts the server
```

Then open <http://127.0.0.1:8000>, click **Choose .AVI File** (or drop a file on the page). The first time a clip is
opened the server computes its depth, with live progress in the banner (about 15 s for a 10 s clip once the model is
loaded; the first request after starting the server is slower, and longer clips take proportionally longer). The same file
opens in a couple of seconds afterwards.

To open a recording straight from your DCIM folder, use `?dcim=<folder>/<file>`, for example
`http://127.0.0.1:8000/?dcim=101NIN03/HNI_0062.AVI`. The server looks for DCIM in `~/Downloads/3DS vids/DCIM`; set
`DCIM_ROOT=/path/to/DCIM` to point it elsewhere (this also applies to `npm run dev`).

### Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | server port |
| `DCIM_ROOT` | `~/Downloads/3DS vids/DCIM` | folder with the 3DS recordings |
| `DA3_RES_W` / `DA3_RES_H` | `252` / `126` | depth model input size; 2:1 and a multiple of 14 |
| `DA3_MAX_VIEWS` | `256` | views per pass (= 128 frames with both eyes); lower it to use less memory |
| `DA3_MAX_GB` | `7` | hard cap on GPU memory; a pass that needs more fails with "out of memory" instead of swapping |
| `DA3_DTYPE` | `bf16` | `bf16`, `fp16` or `fp32` |

With the defaults the GPU peak is about 4.8 GB. Windows of 64 frames (`DA3_MAX_VIEWS=128`) use about 3.7 GB.
Depth results and uploaded files are cached in `server/cache/` (git-ignored, safe to delete).

### Troubleshooting

- **`npm run serve` says the venv is missing:** run `npm run server:setup` first.
- **`uv` / `ffmpeg` not found:** `brew install uv ffmpeg`.
- **No depth appears:** watch the banner; a long clip really does take minutes. Make sure the tab is on the server's
  address (`http://127.0.0.1:8000`), not the dev server, which has no depth server behind it.
- **Out of memory on a long window:** lower `DA3_MAX_VIEWS` (for example `128`).
- **Not on Apple Silicon:** the server falls back to CUDA or CPU. This has not been tested, and CPU is slow.

## Without the server

The viewer also runs DA3 in the browser (WebGPU, per frame) when the server is not running. That path needs ONNX models
in `public/models/` (git-ignored, not included in the repository) and produces noticeably less steady depth, because
each frame is inferred and scaled on its own.

```sh
npm run dev        # Vite dev server
npm run build      # type-check and production build
```

## Tests

```sh
npm run test:unit  # vitest (unit tests, including the depth alignment and flicker helpers)
npm run test:e2e   # Playwright end-to-end suite (starts the dev server)
```

## Tools in `server/`

Run from the repository root with `server/.venv/bin/python`:

- `server/bench_maxviews.py`: how many views fit in one pass for a resolution and precision.
- `server/make_compare.py`: side-by-side videos of the picture and depth from several window sizes
  (`DA3_RES_W=252 DA3_RES_H=126 ... make_compare.py clip.AVI --variants stereo:64,stereo:128`).
- `server/exp_*.py`, `server/demo_clip.py`, `server/run_clip.py`: the experiments behind the defaults (two eyes vs left only,
  3DS1 disparity check, window-size and drift sweeps) and a one-off demo/GLB exporter.
