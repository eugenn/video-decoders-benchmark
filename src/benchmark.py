#!/usr/bin/env python
"""GPU video decoding benchmark: torchcodec vs decord vs PyAV vs PyNvVideoCodec."""
import argparse, ctypes, gc, math, os, statistics, sys, time, traceback
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MEMORY_METRICS = "both"


class _NvmlMemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


_NVML = None
_NVML_HANDLE = None
_NVML_ERROR = None


def _init_nvml():
    global _NVML, _NVML_HANDLE, _NVML_ERROR
    if _NVML_HANDLE is not None or _NVML_ERROR is not None:
        return _NVML_HANDLE is not None
    try:
        _NVML = ctypes.CDLL("libnvidia-ml.so.1")
        rc = _NVML.nvmlInit_v2()
        if rc != 0:
            _NVML_ERROR = f"nvmlInit_v2 failed: {rc}"
            return False
        handle = ctypes.c_void_p()
        device_index = torch.cuda.current_device()
        rc = _NVML.nvmlDeviceGetHandleByIndex_v2(device_index, ctypes.byref(handle))
        if rc != 0:
            _NVML_ERROR = f"nvmlDeviceGetHandleByIndex_v2 failed: {rc}"
            return False
        _NVML_HANDLE = handle
        return True
    except Exception as e:
        _NVML_ERROR = str(e)
        return False


def gpu_device_used_mb():
    """Total used GPU memory from NVML. Includes native CUDA/NVDEC allocations."""
    if not _init_nvml():
        return float("nan")
    info = _NvmlMemoryInfo()
    rc = _NVML.nvmlDeviceGetMemoryInfo(_NVML_HANDLE, ctypes.byref(info))
    if rc != 0:
        return float("nan")
    return info.used / (1024 * 1024)


def gpu_device_delta_mb(base_mb):
    used_mb = gpu_device_used_mb()
    if math.isnan(base_mb) or math.isnan(used_mb):
        return float("nan")
    return max(0.0, used_mb - base_mb)


def measure_torch_memory():
    return MEMORY_METRICS in {"torch", "both"}


def measure_device_memory():
    return MEMORY_METRICS in {"nvml", "both"}


def memory_base_mb():
    return gpu_device_used_mb() if measure_device_memory() else float("nan")


def memory_delta_mb(base_mb):
    return gpu_device_delta_mb(base_mb) if measure_device_memory() else float("nan")


def torch_peak_mb():
    return gpu_mem_snapshot() if measure_torch_memory() else float("nan")


def format_memory_metrics(torch_peak, device_delta):
    parts = []
    if not math.isnan(torch_peak):
        parts.append(f"torch_peak={torch_peak:.1f} MB")
    if not math.isnan(device_delta):
        parts.append(f"gpu_delta={device_delta:.1f} MB")
    return ("  " + "  ".join(parts)) if parts else ""


def nanmedian_or_nan(values):
    clean = [value for value in values if not math.isnan(value)]
    return float(np.median(clean)) if clean else float("nan")


def gpu_mem_snapshot():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_mem():
    if not measure_torch_memory():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# ---------- torchcodec ----------
def bench_torchcodec(video_path, batch_size=2):
    from torchcodec.decoders import VideoDecoder, set_cuda_backend, set_nvdec_cache_capacity
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    set_nvdec_cache_capacity(1)  # Only cache 1 decoder to save memory
    with set_cuda_backend("beta"):
        dec = VideoDecoder(video_path, device="cuda")
        total = dec.metadata.num_frames
        t0 = time.perf_counter()
        count = 0
        # Use smaller batch to avoid massive peak memory spike (53MB per frame in batch)
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            frames = dec.get_frames_in_range(start=batch_start, stop=batch_end)
            count += frames.data.shape[0]
            _ = frames.data.sum()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_torch_mb = torch_peak_mb()
        device_delta_mb = memory_delta_mb(device_base_mb)
        del dec
    return count, elapsed, peak_torch_mb, device_delta_mb


# ---------- PyAV ----------
def bench_pyav(video_path):
    import av
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    container = av.open(
        video_path,
        hwaccel=av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=False),
    )
    stream = container.streams.video[0]
    # FRAME threading is faster for throughput than AUTO/SLICE
    stream.thread_type = "FRAME"
    t0 = time.perf_counter()
    count = 0
    for frame in container.decode(stream):
        # frame is NV12 on host (PyAV copies from GPU).
        # to_ndarray(format="rgb24") is a heavy CPU color conversion bottleneck.
        arr = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(arr).to("cuda", non_blocking=True)
        _ = t.sum()
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_torch_mb = torch_peak_mb()
    device_delta_mb = memory_delta_mb(device_base_mb)
    container.close()
    return count, elapsed, peak_torch_mb, device_delta_mb


def bench_pyav_fast(video_path):
    """Optimized PyAV: Frame threading + NV12 (skips CPU color conversion)."""
    import av
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    container = av.open(
        video_path,
        hwaccel=av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=False),
    )
    stream = container.streams.video[0]
    stream.thread_type = "FRAME"
    t0 = time.perf_counter()
    count = 0
    for frame in container.decode(stream):
        # NV12 is much faster to extract as it avoids the CPU RGB conversion.
        # It produces a Y plane and an interleaved UV plane.
        arr = frame.to_ndarray(format="nv12")
        t = torch.from_numpy(arr).to("cuda", non_blocking=True)
        _ = t.sum()
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_torch_mb = torch_peak_mb()
    device_delta_mb = memory_delta_mb(device_base_mb)
    container.close()
    return count, elapsed, peak_torch_mb, device_delta_mb


_BT601_M = torch.tensor(
    [[1.0,  0.0,        1.402   ],
     [1.0, -0.344136,  -0.714136],
     [1.0,  1.772,      0.0     ]],
    dtype=torch.float32, device="cuda",
)


def _nv12_to_rgb_cuda(nv12: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """NV12 (h*3//2, w) uint8 on GPU -> RGB (h, w, 3) uint8 on GPU. BT.601 limited."""
    y = nv12[:h].to(torch.float32)
    uv = nv12[h:].view(h // 2, w // 2, 2).to(torch.float32) - 128.0
    u = uv[..., 0].repeat_interleave(2, 0).repeat_interleave(2, 1)
    v = uv[..., 1].repeat_interleave(2, 0).repeat_interleave(2, 1)
    yuv = torch.stack([y, u, v], dim=-1)        # (h, w, 3)
    rgb = yuv @ _BT601_M.T
    return rgb.clamp_(0, 255).to(torch.uint8)


def bench_pyav_nv12_gpu_rgb(video_path):
    """PyAV decode in NV12, then YUV->RGB on GPU in torch."""
    import av
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    container = av.open(
        video_path,
        hwaccel=av.codec.hwaccel.HWAccel(device_type="cuda", allow_software_fallback=False),
    )
    stream = container.streams.video[0]
    stream.thread_type = "FRAME"
    h, w = stream.codec_context.height, stream.codec_context.width
    t0 = time.perf_counter()
    count = 0
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="nv12")
        nv12 = torch.from_numpy(arr).to("cuda", non_blocking=True)
        rgb = _nv12_to_rgb_cuda(nv12, h, w)
        _ = rgb.sum()
        count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_torch_mb = torch_peak_mb()
    device_delta_mb = memory_delta_mb(device_base_mb)
    container.close()
    return count, elapsed, peak_torch_mb, device_delta_mb


# ---------- decord ----------
def bench_decord(video_path, batch_size=2):
    import decord
    from decord import VideoReader, gpu
    decord.bridge.set_bridge("torch")
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    vr = VideoReader(video_path, ctx=gpu(0))
    total = len(vr)
    t0 = time.perf_counter()
    count = 0
    for start in range(0, total, batch_size):
        idxs = list(range(start, min(start + batch_size, total)))
        frames = vr.get_batch(idxs)
        count += frames.shape[0]
        _ = frames.sum()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_torch_mb = torch_peak_mb()
    device_delta_mb = memory_delta_mb(device_base_mb)
    del vr
    return count, elapsed, peak_torch_mb, device_delta_mb


# ---------- PyNvVideoCodec ----------
def bench_pynvvideocodec(video_path, batch_size=2):
    import PyNvVideoCodec as nvc
    reset_gpu_mem()
    device_base_mb = memory_base_mb()
    dec = nvc.SimpleDecoder(video_path)
    md = dec.get_stream_metadata()
    total = md.num_frames
    t0 = time.perf_counter()
    count = 0
    while count < total:
        remaining = total - count
        frames = dec.get_batch_frames(min(batch_size, remaining))
        if not frames:
            break
        for f in frames:
            # f is a DecodedFrame; convert to cuda tensor via __cuda_array_interface__
            t = torch.as_tensor(f, device="cuda")
            _ = t.sum()
            count += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_torch_mb = torch_peak_mb()
    device_delta_mb = memory_delta_mb(device_base_mb)
    del dec
    return count, elapsed, peak_torch_mb, device_delta_mb


def make_benches(decord_batch_sizes, torchcodec_batch_size=2, pynv_batch_size=2):
    benches = [
        (
            f"torchcodec (CUDA, batch={torchcodec_batch_size})",
            lambda video_path, bs=torchcodec_batch_size: bench_torchcodec(video_path, bs),
            torchcodec_batch_size,
        ),
        ("PyAV (RGB + FRAME)", bench_pyav, 1),
        ("PyAV (NV12 + FRAME)", bench_pyav_fast, 1),
        ("PyAV (NV12 + GPU RGB)", bench_pyav_nv12_gpu_rgb, 1),
        (
            f"PyNvVideoCodec (NVDEC, batch={pynv_batch_size})",
            lambda video_path, bs=pynv_batch_size: bench_pynvvideocodec(video_path, bs),
            pynv_batch_size,
        ),
    ]
    for batch_size in decord_batch_sizes:
        benches.append((
            f"decord (GPU, batch={batch_size})",
            lambda video_path, bs=batch_size: bench_decord(video_path, bs),
            batch_size,
        ))
    return benches


def run(
    video_path,
    warmup=1,
    runs=3,
    decord_batch_sizes=None,
    torchcodec_batch_size=2,
    pynv_batch_size=2,
):
    video_path = str(Path(video_path).resolve())
    decord_batch_sizes = decord_batch_sizes or [2]
    print(f"\n=== Video: {video_path} ===")
    try:
        import av
        c = av.open(video_path); s = c.streams.video[0]
        print(f"Codec={s.codec_context.name}  {s.width}x{s.height}  "
              f"frames≈{s.frames}  fps={float(s.average_rate):.2f}  "
              f"duration={float(s.duration*s.time_base):.2f}s")
        c.close()
    except Exception as e:
        print("ffprobe failed:", e)

    rows = []
    for name, fn, batch_size in make_benches(
        decord_batch_sizes,
        torchcodec_batch_size=torchcodec_batch_size,
        pynv_batch_size=pynv_batch_size,
    ):
        print(f"\n--- {name} ---")
        try:
            # warmup
            for _ in range(warmup):
                fn(video_path)
            times, torch_peaks, device_deltas, frames_counts = [], [], [], []
            for i in range(runs):
                count, t, torch_peak, device_delta = fn(video_path)
                fps = count / t
                print(
                    f"  run {i+1}: frames={count} time={t:.3f}s  "
                    f"fps={fps:.1f}{format_memory_metrics(torch_peak, device_delta)}"
                )
                times.append(t); torch_peaks.append(torch_peak)
                device_deltas.append(device_delta); frames_counts.append(count)
            rows.append({
                "library": name,
                "batch_size": batch_size,
                "frames_decoded": int(np.median(frames_counts)),
                "median_time_s": float(np.median(times)),
                "median_fps": float(np.median(frames_counts) / np.median(times)),
                "min_time_s": float(min(times)),
                "max_time_s": float(max(times)),
                "peak_torch_alloc_mb": nanmedian_or_nan(torch_peaks),
                "gpu_mem_delta_mb": nanmedian_or_nan(device_deltas),
                "status": "ok",
            })
        except Exception as e:
            traceback.print_exc()
            rows.append({
                "library": name, "batch_size": batch_size,
                "frames_decoded": 0, "median_time_s": float("nan"),
                "median_fps": 0.0, "min_time_s": float("nan"), "max_time_s": float("nan"),
                "peak_torch_alloc_mb": 0.0, "gpu_mem_delta_mb": float("nan"),
                "status": f"error: {type(e).__name__}: {str(e)[:120]}",
            })
    return rows


def plot(rows, video_name):
    df = pd.DataFrame(rows).sort_values("median_fps", ascending=False)
    ok = df[df.status == "ok"]
    names = ok.library.tolist()
    has_memory = ok.gpu_mem_delta_mb.notna().any() or ok.peak_torch_alloc_mb.notna().any()

    fig, axes = plt.subplots(1, 3 if has_memory else 2, figsize=(18 if has_memory else 12, 5))

    # 1: FPS
    ax = axes[0]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6", "#f97316", "#64748b"]
    bars = ax.bar(names, ok.median_fps, color=colors[:len(names)])
    ax.set_ylabel("Frames per second"); ax.set_title(f"Decode throughput (higher = faster)\n{video_name}")
    ax.tick_params(axis="x", rotation=15)
    for b,v in zip(bars, ok.median_fps): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom")

    # 2: time
    ax = axes[1]
    ms_per_frame = ok.median_time_s / ok.frames_decoded * 1000
    bars = ax.bar(names, ms_per_frame, color=colors[:len(names)])
    ax.set_ylabel("ms per frame"); ax.set_title("Time per frame (lower = faster)")
    ax.tick_params(axis="x", rotation=15)
    for b,v in zip(bars, ms_per_frame): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", va="bottom")

    if has_memory:
        # 3: memory
        ax = axes[2]
        mem_col = "gpu_mem_delta_mb" if ok.gpu_mem_delta_mb.notna().any() else "peak_torch_alloc_mb"
        bars = ax.bar(names, ok[mem_col], color=colors[:len(names)])
        ax.set_ylabel("GPU memory delta (MB)" if mem_col == "gpu_mem_delta_mb" else "Peak torch alloc (MB)")
        ax.set_title("GPU memory")
        ax.tick_params(axis="x", rotation=15)
        for b,v in zip(bars, ok[mem_col]): ax.text(b.get_x()+b.get_width()/2, v, f"{v:.0f}", ha="center", va="bottom")

    fig.suptitle(f"GPU Video Decoding Benchmark — Tesla T4 — {video_name}", fontsize=13, y=1.02)
    fig.tight_layout()
    png = RESULTS_DIR / "benchmark.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"\nSaved plot: {png}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--torchcodec-batch-size", type=int, default=2,
                    help="Batch size for torchcodec get_frames_in_range")
    ap.add_argument("--pynv-batch-size", type=int, default=2,
                    help="Batch size for PyNvVideoCodec get_batch_frames")
    ap.add_argument("--decord-batch-sizes", default="2",
                    help="Comma-separated decord batch sizes to benchmark")
    ap.add_argument("--memory-metrics", choices=["both", "torch", "nvml", "none"], default="both",
                    help="Memory metrics to collect. Use 'none' for FPS-only timing")
    args = ap.parse_args()

    global MEMORY_METRICS
    MEMORY_METRICS = args.memory_metrics

    decord_batch_sizes = [int(x) for x in args.decord_batch_sizes.split(",") if x]

    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
    if measure_device_memory() and not _init_nvml():
        print(f"NVML memory metric unavailable: {_NVML_ERROR}")
    rows = run(
        args.video,
        warmup=args.warmup,
        runs=args.runs,
        decord_batch_sizes=decord_batch_sizes,
        torchcodec_batch_size=args.torchcodec_batch_size,
        pynv_batch_size=args.pynv_batch_size,
    )
    video_name = Path(args.video).name
    df = plot(rows, video_name)
    csv = RESULTS_DIR / "benchmark.csv"
    summary_df = pd.DataFrame(rows)
    if MEMORY_METRICS == "none":
        summary_df = summary_df.drop(columns=["peak_torch_alloc_mb", "gpu_mem_delta_mb"])
    summary_df.to_csv(csv, index=False)
    print(f"Saved csv:  {csv}")
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
