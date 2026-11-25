"""Analyze SES experiment results: summary statistics, CSV export, and NRE scaling plots.

Usage examples:
    python analyze.py --out_dir ./ses_results
    python analyze.py --out_dir ./ses_results_nre200 --baseline "SES(CEM)" --grid_step 10
"""
import argparse
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_results(raw_results: dict):
    """Support both new dict format {seed: score} and legacy list format [s0, s1]."""
    norm = {}
    for key, val in raw_results.items():
        if isinstance(val, dict):
            norm[key] = {str(k): float(v) for k, v in val.items()}
        elif isinstance(val, list):
            norm[key] = {str(i): float(v) for i, v in enumerate(val)}
    return norm


def eval_trajectory_on_grid(history, grid: np.ndarray) -> np.ndarray:
    """Step-function evaluation of a monotonic (nre, best_score) trajectory on a common NRE grid."""
    if not history:
        return np.full(len(grid), np.nan)
    nres = np.array([h[0] for h in history], dtype=float)
    scores = np.array([h[1] for h in history], dtype=float)
    idx = np.searchsorted(nres, grid, side="right") - 1
    return np.where(idx >= 0, scores[np.clip(idx, 0, len(scores) - 1)], np.nan)


def print_and_save_overall(results, methods, prompts, baseline, analysis_dir):
    rows = []
    # Gather paired scores for baseline comparison
    base_map = {}
    if baseline in methods:
        for p in prompts:
            for s, sc in results.get(f"{baseline}|{p}", {}).items():
                base_map[(p, s)] = sc

    print("\n================ Overall Summary (across all prompts & seeds) ================")
    header = f"{'Method':16s} | {'Mean +- Std':15s} | {'Median':7s} | {'Min':6s} | {'Max':6s} | {'Delta vs ' + baseline:16s} | {'WinRate':8s} | {'n':3s}"
    print(header)
    print("-" * len(header))

    for m in methods:
        scores = []
        deltas = []
        wins = 0
        paired = 0
        for p in prompts:
            for s, sc in results.get(f"{m}|{p}", {}).items():
                scores.append(sc)
                if (p, s) in base_map:
                    paired += 1
                    diff = sc - base_map[(p, s)]
                    deltas.append(diff)
                    if diff > 1e-9:
                        wins += 1

        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        mean_v, std_v = float(arr.mean()), float(arr.std())
        med_v, min_v, max_v = float(np.median(arr)), float(arr.min()), float(arr.max())

        if m == baseline or paired == 0:
            delta_str = "-"
            win_str = "-"
            mean_delta, win_rate = np.nan, np.nan
        else:
            mean_delta = float(np.mean(deltas))
            win_rate = 100.0 * wins / paired
            delta_str = f"{mean_delta:+.2f}"
            win_str = f"{win_rate:5.1f}%"

        print(
            f"{m:16s} | {mean_v:6.2f} +- {std_v:<5.2f} | {med_v:7.2f} | "
            f"{min_v:6.2f} | {max_v:6.2f} | {delta_str:16s} | {win_str:8s} | {len(arr):3d}"
        )
        rows.append(
            {
                "method": m,
                "mean": round(mean_v, 4),
                "std": round(std_v, 4),
                "median": round(med_v, 4),
                "min": round(min_v, 4),
                "max": round(max_v, 4),
                f"delta_vs_{baseline}": "" if np.isnan(mean_delta) else round(mean_delta, 4),
                f"winrate_vs_{baseline}_pct": "" if np.isnan(win_rate) else round(win_rate, 2),
                "n": len(arr),
            }
        )

    csv_path = os.path.join(analysis_dir, "summary_overall.csv")
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[Saved] {csv_path}")


def print_and_save_per_prompt(results, methods, prompts, analysis_dir):
    print("\n================ Per-Prompt Breakdown (mean +- std over seeds) ================")
    col_w = 15
    header = f"{'Prompt':38s} | " + " | ".join(f"{m:{col_w}s}" for m in methods)
    print(header)
    print("-" * len(header))

    csv_rows = []
    for p in prompts:
        cells = []
        row_dict = {"prompt": p}
        for m in methods:
            vals = list(results.get(f"{m}|{p}", {}).values())
            if vals:
                mean_v, std_v = float(np.mean(vals)), float(np.std(vals))
                cells.append(f"{mean_v:6.2f} +- {std_v:<5.2f}")
                row_dict[f"{m}_mean"] = round(mean_v, 4)
                row_dict[f"{m}_std"] = round(std_v, 4)
            else:
                cells.append(f"{'N/A':^{col_w}s}")
                row_dict[f"{m}_mean"] = ""
                row_dict[f"{m}_std"] = ""
        short_p = (p[:35] + "...") if len(p) > 38 else p
        print(f"{short_p:38s} | " + " | ".join(cells))
        csv_rows.append(row_dict)

    csv_path = os.path.join(analysis_dir, "summary_per_prompt.csv")
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved] {csv_path}")


def plot_scaling_curves(histories, methods, prompts, max_nre, grid_step, analysis_dir):
    grid = np.arange(grid_step, max_nre + 1, grid_step, dtype=float)
    if grid.size == 0 or grid[-1] < max_nre:
        grid = np.append(grid, float(max_nre))

    # 1. Overall scaling curve (mean +- std across all prompts & seeds)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in methods:
        trajs = []
        for p in prompts:
            for hist in histories.get(f"{m}|{p}", {}).values():
                trajs.append(eval_trajectory_on_grid(hist, grid))
        if not trajs:
            continue
        mat = np.vstack(trajs)
        mean_curve = np.nanmean(mat, axis=0)
        std_curve = np.nanstd(mat, axis=0)
        (line,) = ax.plot(grid, mean_curve, marker="o", markersize=3.5, linewidth=2, label=m)
        ax.fill_between(
            grid,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=0.15,
        )

    ax.set_xlabel("NRE (Number of Reward Evaluations)", fontsize=11)
    ax.set_ylabel("Best Reward Score (mean ± std)", fontsize=11)
    ax.set_title("Inference-Time Scaling Curve", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    plt.tight_layout()
    overall_plot = os.path.join(analysis_dir, "scaling_curve.png")
    plt.savefig(overall_plot, dpi=150)
    plt.close(fig)
    print(f"[Saved] {overall_plot}")

    # 2. Per-prompt scaling curves
    n_p = len(prompts)
    ncols = min(3, n_p)
    nrows = (n_p + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    for idx, p in enumerate(prompts):
        ax = axes[idx // ncols][idx % ncols]
        for m in methods:
            trajs = [
                eval_trajectory_on_grid(hist, grid)
                for hist in histories.get(f"{m}|{p}", {}).values()
            ]
            if not trajs:
                continue
            mat = np.vstack(trajs)
            mean_curve = np.nanmean(mat, axis=0)
            std_curve = np.nanstd(mat, axis=0)
            (line,) = ax.plot(grid, mean_curve, linewidth=1.8, label=m)
            ax.fill_between(
                grid,
                mean_curve - std_curve,
                mean_curve + std_curve,
                color=line.get_color(),
                alpha=0.15,
            )
        ax.set_title(p[:42] + ("..." if len(p) > 42 else ""), fontsize=9)
        ax.set_xlabel("NRE", fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.35)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n_p, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    per_prompt_plot = os.path.join(analysis_dir, "scaling_curves_per_prompt.png")
    plt.savefig(per_prompt_plot, dpi=150)
    plt.close(fig)
    print(f"[Saved] {per_prompt_plot}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SES experiment outputs")
    parser.add_argument("--out_dir", type=str, default="./ses_results", help="Experiment directory")
    parser.add_argument("--baseline", type=str, default="BoN", help="Baseline method for Delta / WinRate")
    parser.add_argument("--grid_step", type=int, default=10, help="NRE step size for scaling curves")
    return parser.parse_args()


def main():
    args = parse_args()
    res_path = os.path.join(args.out_dir, "results.json")
    hist_path = os.path.join(args.out_dir, "histories.json")
    meta_path = os.path.join(args.out_dir, "meta.json")

    raw_results = load_json(res_path)
    if not raw_results:
        raise FileNotFoundError(f"Could not find or load {res_path}")
    results = normalize_results(raw_results)
    histories = load_json(hist_path)
    meta = load_json(meta_path) or {}

    # Preserve method & prompt order from meta.json if available
    methods = meta.get("methods") or list(dict.fromkeys(k.split("|")[0] for k in results))
    prompts = meta.get("prompts") or list(dict.fromkeys(k.split("|", 1)[1] for k in results))

    analysis_dir = os.path.join(args.out_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    print_and_save_overall(results, methods, prompts, args.baseline, analysis_dir)
    print_and_save_per_prompt(results, methods, prompts, analysis_dir)

    if histories:
        max_nre = meta.get("nre") or max(
            int(h[-1][0])
            for m_dict in histories.values()
            for h in m_dict.values()
            if h
        )
        plot_scaling_curves(histories, methods, prompts, max_nre, args.grid_step, analysis_dir)
    else:
        print("\n[Info] histories.json not found; skipping NRE scaling curve plots.")


if __name__ == "__main__":
    main()