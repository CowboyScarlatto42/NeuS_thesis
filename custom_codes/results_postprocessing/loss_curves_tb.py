#!/usr/bin/env python3
"""
Plot NeuS TensorBoard scalars (4 must-have curves) without launching TensorBoard UI.

Also supports timing estimation from TensorBoard event wall_time:
- elapsed wall-time (start -> end)
- median/mean sec per iteration
- iterations per hour

Example:
python tb_curves_neus.py \
  --logdir "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/logs" \
  --out_dir "/content/drive/MyDrive/Tesi/neus/serious_analysis/runs/hst_black_nmc_32/post/tb" \
  --ema_alpha 0.05 \
  --timing \
  --export_csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


LOSS_TAGS_DEFAULT = ["Loss/loss", "Loss/color_loss", "Loss/eikonal_loss"]
PSNR_TAG_DEFAULT = "Statistics/psnr"
TIMING_TAG_DEFAULT = "Loss/loss"


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float64)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def load_scalar(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
    evs = ea.Scalars(tag)
    steps = np.array([e.step for e in evs], dtype=int)
    vals = np.array([e.value for e in evs], dtype=float)
    return steps, vals


def load_scalar_with_time(ea: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return steps, values, wall_time (seconds since epoch)."""
    evs = ea.Scalars(tag)
    steps = np.array([e.step for e in evs], dtype=int)
    vals = np.array([e.value for e in evs], dtype=float)
    times = np.array([e.wall_time for e in evs], dtype=float)
    return steps, vals, times


def save_plot(fig, out_path: Path | None, dpi: int):
    if out_path is None:
        plt.show()
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"[Saved] {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=Path, required=True, help="Directory containing events.out.tfevents*")
    p.add_argument("--out_dir", type=Path, default=None, help="If set, save PNGs here. If omitted, only show.")
    p.add_argument("--ema_alpha", type=float, default=0.05, help="EMA smoothing factor.")
    p.add_argument("--dpi", type=int, default=200, help="DPI for saved figures.")
    p.add_argument("--export_csv", action="store_true", help="Export raw+EMA values as CSV in out_dir.")
    p.add_argument("--loss_tags", nargs="+", default=LOSS_TAGS_DEFAULT, help="Loss tags to plot.")
    p.add_argument("--psnr_tag", default=PSNR_TAG_DEFAULT, help="PSNR tag to plot.")

    # timing options
    p.add_argument("--timing", action="store_true", help="Estimate training time from TensorBoard wall_time.")
    p.add_argument(
        "--timing_tag",
        default=TIMING_TAG_DEFAULT,
        help="Scalar tag used to estimate timing (should be logged frequently).",
    )
    p.add_argument(
        "--timing_gap_threshold_sec",
        type=float,
        default=600.0,
        help="Filter out pauses: ignore dt > threshold (seconds). Set <=0 to disable filtering.",
    )
    p.add_argument(
        "--save_timing_samples",
        action="store_true",
        help="If --out_dir is set, save per-interval timing samples as CSV.",
    )
    return p.parse_args()


def compute_timing(steps: np.ndarray, times: np.ndarray, gap_threshold_sec: float):
    # sort by step
    idx = np.argsort(steps)
    steps = steps[idx]
    times = times[idx]

    # total elapsed
    total_sec = float(times[-1] - times[0])

    dstep = np.diff(steps)
    dt = np.diff(times)

    mask = dstep > 0
    if gap_threshold_sec is not None and gap_threshold_sec > 0:
        mask = mask & (dt <= gap_threshold_sec)

    sec_per_iter = dt[mask] / dstep[mask]

    # handle edge cases
    if sec_per_iter.size == 0:
        stats = {
            "elapsed_sec": total_sec,
            "elapsed_hr": total_sec / 3600.0,
            "sec_per_iter_median": float("nan"),
            "sec_per_iter_mean": float("nan"),
            "iters_per_hour_median": float("nan"),
            "n_intervals_used": 0,
            "n_intervals_total": int(len(dt)),
        }
    else:
        med = float(np.median(sec_per_iter))
        mean = float(np.mean(sec_per_iter))
        stats = {
            "elapsed_sec": total_sec,
            "elapsed_hr": total_sec / 3600.0,
            "sec_per_iter_median": med,
            "sec_per_iter_mean": mean,
            "iters_per_hour_median": (3600.0 / med) if med > 0 else float("inf"),
            "n_intervals_used": int(sec_per_iter.size),
            "n_intervals_total": int(len(dt)),
        }

    samples = {
        "step_prev": steps[:-1],
        "step_next": steps[1:],
        "dstep": dstep,
        "t_prev": times[:-1],
        "t_next": times[1:],
        "dt_sec": dt,
        "used": mask.astype(int),
    }
    # Only define sec_per_iter where used==1; else NaN (for easy plotting/analysis)
    sec_per_iter_full = np.full_like(dt, np.nan, dtype=np.float64)
    sec_per_iter_full[mask] = dt[mask] / dstep[mask]
    samples["sec_per_iter"] = sec_per_iter_full

    return stats, samples


def main():
    args = parse_args()

    if not args.logdir.exists():
        raise FileNotFoundError(f"logdir not found: {args.logdir}")

    ea = EventAccumulator(str(args.logdir), size_guidance={"scalars": 0})
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    print("Found scalar tags:")
    for t in tags:
        print(" -", t)

    # validate plot tags
    missing = [t for t in (args.loss_tags + [args.psnr_tag]) if t not in tags]
    if missing:
        raise ValueError(f"Missing expected tags: {missing}\nAvailable: {tags}")

    # --- Load data for plots
    loss_data = {}
    for t in args.loss_tags:
        s, v = load_scalar(ea, t)
        loss_data[t] = (s, v, ema(v, args.ema_alpha))

    ps_s, ps_v = load_scalar(ea, args.psnr_tag)
    ps_ema = ema(ps_v, args.ema_alpha)

    # --- Plot 1: losses raw (log)
    fig = plt.figure(figsize=(9, 5))
    for t, (s, v, _) in loss_data.items():
        plt.plot(s, v, label=t)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot(fig, (args.out_dir / "losses_raw.png") if args.out_dir else None, args.dpi)
    plt.close(fig)

    # --- Plot 2: losses EMA (log)
    fig = plt.figure(figsize=(9, 5))
    for t, (s, _, ve) in loss_data.items():
        plt.plot(s, ve, label=f"{t} (EMA)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (smoothed)")
    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot(fig, (args.out_dir / "losses_ema.png") if args.out_dir else None, args.dpi)
    plt.close(fig)

    # --- Plot 3: PSNR raw
    fig = plt.figure(figsize=(9, 4))
    plt.plot(ps_s, ps_v, label=args.psnr_tag)
    plt.xlabel("Iteration")
    plt.ylabel("PSNR")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot(fig, (args.out_dir / "psnr_raw.png") if args.out_dir else None, args.dpi)
    plt.close(fig)

    # --- Plot 4: PSNR EMA
    fig = plt.figure(figsize=(9, 4))
    plt.plot(ps_s, ps_ema, label=f"{args.psnr_tag} (EMA)")
    plt.xlabel("Iteration")
    plt.ylabel("PSNR (smoothed)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    save_plot(fig, (args.out_dir / "psnr_ema.png") if args.out_dir else None, args.dpi)
    plt.close(fig)

    # --- Optional CSV export (curves)
    if args.export_csv:
        if args.out_dir is None:
            raise ValueError("--export_csv requires --out_dir.")

        out = args.out_dir
        out.mkdir(parents=True, exist_ok=True)

        for t, (s, v, ve) in loss_data.items():
            df = pd.DataFrame({"step": s, "value": v, "ema": ve})
            csv_path = out / f"{t.replace('/', '_')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"[Saved] {csv_path}")

        df = pd.DataFrame({"step": ps_s, "value": ps_v, "ema": ps_ema})
        csv_path = out / f"{args.psnr_tag.replace('/', '_')}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")

    # --- Timing estimation
    if args.timing:
        if args.timing_tag not in tags:
            raise ValueError(f"--timing_tag '{args.timing_tag}' not found. Available: {tags}")

        t_steps, _, t_times = load_scalar_with_time(ea, args.timing_tag)
        if len(t_steps) < 5:
            raise ValueError(f"Not enough events for timing tag '{args.timing_tag}' (n={len(t_steps)}).")

        stats, samples = compute_timing(t_steps, t_times, args.timing_gap_threshold_sec)

        print("\n=== Timing (from TensorBoard wall_time) ===")
        print(f"Tag used          : {args.timing_tag}")
        print(f"Step range        : {int(t_steps.min())} -> {int(t_steps.max())}")
        print(f"Elapsed wall-time : {stats['elapsed_sec']:.1f} s  ({stats['elapsed_hr']:.2f} h)")
        print(f"Intervals used    : {stats['n_intervals_used']} / {stats['n_intervals_total']} "
              f"(gap_threshold={args.timing_gap_threshold_sec}s)")
        print(f"Median sec/iter   : {stats['sec_per_iter_median']:.4f}")
        print(f"Mean sec/iter     : {stats['sec_per_iter_mean']:.4f}")
        print(f"Iters/hour (med)  : {stats['iters_per_hour_median']:.1f}")

        if args.out_dir is not None:
            out = args.out_dir
            out.mkdir(parents=True, exist_ok=True)

            # save a small text summary
            timing_txt = out / "timing.txt"
            with open(timing_txt, "w") as f:
                for k, v in stats.items():
                    f.write(f"{k}: {v}\n")
                f.write(f"timing_tag: {args.timing_tag}\n")
                f.write(f"gap_threshold_sec: {args.timing_gap_threshold_sec}\n")
            print(f"[Saved] {timing_txt}")

            if args.save_timing_samples:
                df = pd.DataFrame(samples)
                timing_csv = out / "timing_samples.csv"
                df.to_csv(timing_csv, index=False)
                print(f"[Saved] {timing_csv}")


if __name__ == "__main__":
    main()
