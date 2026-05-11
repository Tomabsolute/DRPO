#!/usr/bin/env python3
import argparse
import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log_file(path: Path):
    values = []
    epochs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s.startswith("{") or "'is_ratio'" not in s:
                continue
            try:
                obj = ast.literal_eval(s)
            except Exception:
                continue
            v = obj.get("is_ratio")
            e = obj.get("epoch")
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isfinite(fv):
                continue
            values.append(fv)
            try:
                fe = float(e)
            except Exception:
                fe = float("nan")
            epochs.append(fe)
    return values, epochs


def moving_average(xs, window):
    if window <= 1 or len(xs) == 0:
        return xs
    out = []
    run = 0.0
    for i, x in enumerate(xs):
        run += x
        if i >= window:
            run -= xs[i - window]
        n = min(i + 1, window)
        out.append(run / n)
    return out


def main():
    parser = argparse.ArgumentParser(description="Visualize is_ratio from training log")
    parser.add_argument("log_file", help="Path to training log file")
    parser.add_argument("--out", default="is_ratio.png", help="Output image path")
    parser.add_argument("--ma-window", type=int, default=20, help="Moving average window")
    parser.add_argument("--use-epoch", action="store_true", help="Use epoch as x-axis when available")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    values, epochs = parse_log_file(log_path)
    if not values:
        raise SystemExit(f"No finite is_ratio found in {log_path}")

    x = list(range(1, len(values) + 1))
    xlabel = "Log Record Index"
    if args.use_epoch and any(math.isfinite(e) for e in epochs):
        x = [e if math.isfinite(e) else i + 1 for i, e in enumerate(epochs)]
        xlabel = "Epoch"

    ma = moving_average(values, args.ma_window)

    plt.figure(figsize=(10, 5))
    plt.plot(x, values, linewidth=1.2, alpha=0.6, label="is_ratio")
    if args.ma_window > 1:
        plt.plot(x, ma, linewidth=2.0, label=f"MA({args.ma_window})")

    plt.axhline(1.0, linestyle="--", linewidth=1, color="gray", alpha=0.8, label="ratio=1")
    plt.xlabel(xlabel)
    plt.ylabel("is_ratio")
    plt.title(f"is_ratio Trend ({log_path.name})")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180)

    print(f"Parsed {len(values)} points")
    print(f"min={min(values):.6f}, max={max(values):.6f}, mean={sum(values)/len(values):.6f}")
    print(f"Saved plot to: {args.out}")


if __name__ == "__main__":
    main()
