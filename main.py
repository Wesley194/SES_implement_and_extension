"""Main experiment runner for Spectral Evolution Search benchmarks.

Outputs saved under --out_dir:
    - meta.json             : experiment configuration
    - results.json          : best score per (method|prompt, seed)
    - histories.json        : per-round (nre, best_score) trajectory per (method|prompt, seed)
    - noises/*.npy          : best initial noise x0 for each run
    - images/*.png          : generated best images & gallery_seed{s}.png

Usage examples:
    CUDA_VISIBLE_DEVICES=5 python main.py --smoke
    CUDA_VISIBLE_DEVICES=5 python main.py --nre 200 --seeds 0 1 2 --out_dir ./ses_results_nre200
"""
import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms import ALGORITHMS, get_algorithm
from models import CLIPReward, SD15Generator, TaskEvaluator

DEFAULT_PROMPTS = [
    "A black colored car.",
    "An orange colored sandwich.",
    "A stack of 3 plates. A blue plate is on the top, a green plate is in the middle, a blue plate is on the bottom.",
    "A red colored dog.",
    "A photo of a wine glass on a wooden table.",
    "A cat wearing a small hat, studio lighting.",
    "A blue backpack on the left of a yellow umbrella.",
    "Four apples arranged in a row on a marble counter.",
    "A ceramic teapot shaped like an elephant, warm morning light.",
    "An astronaut riding a horse on Mars, photorealistic.",
]

DEFAULT_METHODS = ["BoN", "SES(CEM)", "SES-CMA", "SES-CMA-diag", "SES-GNN"]


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:60].strip("_")


def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def save_json(path: str, data: dict):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def run_determinism_check(evaluator: TaskEvaluator, img_dir: str) -> float:
    print("\n--- [1] Determinism & Speed Check ---", flush=True)
    prompt = "A black colored car."
    x0 = np.random.default_rng(0).standard_normal(evaluator.latent_shape)
    img_a = evaluator.generate(prompt, x0)
    img_b = evaluator.generate(prompt, x0)
    diff = np.abs(np.asarray(img_a, dtype=float) - np.asarray(img_b, dtype=float)).max()
    print(f"max pixel diff for the same noise: {diff:.6f} (should be ~0)", flush=True)

    t0 = time.time()
    for i in range(5):
        evaluator.generate(
            prompt, np.random.default_rng(i).standard_normal(evaluator.latent_shape)
        )
    sec_per_img = (time.time() - t0) / 5.0
    print(f"seconds per image: {sec_per_img:.3f}s", flush=True)

    check_path = os.path.join(img_dir, "determinism_check_sample.png")
    img_a.save(check_path)
    print(f"Saved check sample to: {check_path}", flush=True)
    return sec_per_img


def run_smoke_test(evaluator: TaskEvaluator, method_names, level: int, nre: int = 20):
    print(f"\n--- [2] Smoke Test (1 prompt, NRE={nre}, level={level}) ---", flush=True)
    obj = evaluator.make_objective("A black colored car.")
    for name in method_names:
        algo = get_algorithm(name)
        res = algo(obj, evaluator.latent_shape, budget=nre, level=level, seed=0)
        print(
            f"{name:16s} best={res.best_score:.4f}  (logged {len(res.history)} steps)",
            flush=True,
        )


def run_full_experiment(
    evaluator: TaskEvaluator,
    prompts,
    method_names,
    nre: int,
    seeds,
    level: int,
    use_lcm: bool,
    out_dir: str,
):
    noise_dir = os.path.join(out_dir, "noises")
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    def noise_path(m, p, s):
        return os.path.join(noise_dir, f"{safe_name(m)}__{safe_name(p)}__s{s}.npy")

    res_path = os.path.join(out_dir, "results.json")
    hist_path = os.path.join(out_dir, "histories.json")
    meta_path = os.path.join(out_dir, "meta.json")

    results = load_json(res_path)
    histories = load_json(hist_path)

    save_json(
        meta_path,
        {
            "nre": nre,
            "seeds": list(seeds),
            "level": level,
            "use_lcm": use_lcm,
            "methods": list(method_names),
            "prompts": list(prompts),
        },
    )

    print(
        f"\n--- [3] Full Experiment (prompts={len(prompts)}, methods={method_names}, "
        f"NRE={nre}, seeds={seeds}, level={level}) ---",
        flush=True,
    )
    for p in prompts:
        obj = evaluator.make_objective(p)
        for m in method_names:
            key = f"{m}|{p}"
            results.setdefault(key, {})
            histories.setdefault(key, {})
            algo = get_algorithm(m)

            for s in seeds:
                sk = str(s)
                npath = noise_path(m, p, s)
                if (
                    sk in results[key]
                    and sk in histories[key]
                    and os.path.exists(npath)
                ):
                    print(
                        f"  [Skip cached] {m:15s} | seed={s} | score={results[key][sk]:.2f} | {p[:40]}",
                        flush=True,
                    )
                    continue

                t0 = time.time()
                res = algo(obj, evaluator.latent_shape, budget=nre, level=level, seed=s)

                # Save best initial noise
                np.save(npath, np.asarray(res.best_x0, dtype=np.float32))
                # Save best score & per-round NRE history
                results[key][sk] = float(res.best_score)
                histories[key][sk] = [[int(n), float(sc)] for n, sc in res.history]

                save_json(res_path, results)
                save_json(hist_path, histories)

                print(
                    f"  done: {m:15s} | seed={s} | score={res.best_score:.2f} | "
                    f"steps={len(res.history):3d} | {time.time()-t0:.1f}s | {p[:40]}",
                    flush=True,
                )

    # Save individual images & gallery for each seed
    print(f"\n--- [4] Generating Images & Galleries into {img_dir} ---", flush=True)
    cols = ["default"] + list(method_names)
    for show_seed in seeds:
        fig, axes = plt.subplots(
            len(prompts),
            len(cols),
            figsize=(3 * len(cols), 3.3 * len(prompts)),
            squeeze=False,
        )
        for i, p in enumerate(prompts):
            for j, m in enumerate(cols):
                ax = axes[i][j]
                ax.axis("off")
                if m == "default":
                    x0 = np.random.default_rng(1000 + show_seed).standard_normal(
                        evaluator.latent_shape
                    )
                else:
                    fp = noise_path(m, p, show_seed)
                    if not os.path.exists(fp):
                        ax.set_title(f"{m}\n(no saved noise)", fontsize=8)
                        continue
                    x0 = np.load(fp).astype(np.float64)

                img = evaluator.generate(p, x0)
                sc = evaluator.score(p, img)
                img_file = os.path.join(
                    img_dir, f"{safe_name(m)}__{safe_name(p)}__s{show_seed}.png"
                )
                img.save(img_file)
                ax.imshow(img)
                head = (p[:32] + "...\n") if j == 0 else ""
                ax.set_title(f"{head}{m}  CLIP {sc:.2f}", fontsize=8)

        plt.tight_layout()
        gallery_path = os.path.join(img_dir, f"gallery_seed{show_seed}.png")
        plt.savefig(gallery_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved gallery: {gallery_path}", flush=True)

    print(
        f"\n[Done] Run analysis with:\n  python analyze.py --out_dir {out_dir}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Spectral Evolution Search Benchmark")
    parser.add_argument("--out_dir", type=str, default="./ses_results", help="Output directory")
    parser.add_argument("--nre", type=int, default=100, help="Number of reward evaluations per run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="Random seeds")
    parser.add_argument("--level", type=int, default=3, help="Wavelet decomposition level J")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=DEFAULT_METHODS,
        choices=list(ALGORITHMS.keys()),
        help="Algorithms to benchmark",
    )
    parser.add_argument("--use_lcm", action="store_true", help="Use 4-step LCM-LoRA proxy")
    parser.add_argument("--smoke", action="store_true", help="Only run determinism check + smoke test")
    parser.add_argument("--skip_check", action="store_true", help="Skip determinism & speed check")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(
        f"[GPU] torch.cuda.is_available={torch.cuda.is_available()}, "
        f"count={torch.cuda.device_count()}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(f"[GPU] Current device: {torch.cuda.get_device_name(0)}", flush=True)

    generator = SD15Generator(use_lcm=args.use_lcm)
    reward_model = CLIPReward()
    evaluator = TaskEvaluator(generator, reward_model)

    if not args.skip_check:
        sec_per_img = run_determinism_check(evaluator, img_dir)
        if not args.smoke:
            total_imgs = len(DEFAULT_PROMPTS) * len(args.methods) * len(args.seeds) * args.nre
            est_min = total_imgs * sec_per_img / 60.0
            print(f"[Estimate] Full run = {total_imgs} images (~{est_min:.1f} min)", flush=True)

    if args.smoke:
        run_smoke_test(evaluator, args.methods, level=args.level, nre=20)
        return

    run_full_experiment(
        evaluator=evaluator,
        prompts=DEFAULT_PROMPTS,
        method_names=args.methods,
        nre=args.nre,
        seeds=tuple(args.seeds),
        level=args.level,
        use_lcm=args.use_lcm,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()