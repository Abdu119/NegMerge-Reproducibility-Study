"""N-variation ablation for NegMerge reproducibility study.

Sweeps subset size N in [3, 5, 10, 15, 20, 25, 30] and measures how the
sign-consensus rate and post-merge accuracies degrade as N shrinks.
Writes per-(N, seed) records to results/n_variation.json (resumable) and a
flat summary to results/n_variation.csv. Runs the full N=30 validation
before proceeding to smaller N.
"""

import os
import sys
import json
import csv
import random
import argparse
import time

import torch
import numpy as np

# ----------------------------------------------------------------------
# MaybeToTensor fix — must be applied before any timm-using imports.
from torchvision.transforms.functional import to_tensor
import timm.data.transforms


class MaybeToTensor:
    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return to_tensor(x)


timm.data.transforms.MaybeToTensor = MaybeToTensor

# ----------------------------------------------------------------------
# Path bootstrap so `from src...` works regardless of CWD, AND so the local
# `src/datasets/` package does not shadow HuggingFace `datasets` when the
# script is invoked as `python src/eval_n_variation.py` (Python would otherwise
# prepend src/ to sys.path).
_THIS_FILE = os.path.abspath(__file__)
_SCRIPT_DIR = os.path.dirname(_THIS_FILE)
BASE_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path[:] = [p for p in sys.path if os.path.abspath(p) != _SCRIPT_DIR]
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.modeling import ImageEncoder
from src.task_vectors import NonLinearTaskVector
from src.eval import evaluate_task_vector_at_coef

# ----------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints/standard/ViT-B-32")
PRETRAINED_PATH = os.path.join(CHECKPOINT_DIR, "zeroshot.pt")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
JSON_OUT = os.path.join(RESULTS_DIR, "n_variation.json")
CSV_OUT = os.path.join(RESULTS_DIR, "n_variation.csv")

N_VALUES = [3, 5, 10, 15, 20, 25, 30]
ALPHAS = [0.50, 0.75, 0.90, 0.95, 1.00]
SEEDS_SMALL_N = [0, 1, 2]
SEEDS_N30 = [0]
IMAGENET_THRESHOLD = 0.5925  # 0.95 * 62.37%

# Baseline values the N=30, seed=0 validation must hit (within tolerance).
BASELINE_CONSENSUS = 0.0966
BASELINE_CARS_AT_095 = 0.2408
BASELINE_IMAGENET_AT_095 = 0.5927
TOL_CONSENSUS = 0.0001  # 0.01 percentage-points
TOL_ACCURACY = 0.001    # 0.1 percentage-points

# Enumerate the 30 checkpoints in the same order as the baseline run.
FINETUNED_PATHS = []
for m in range(1, 11):
    for n in range(1, 4):
        FINETUNED_PATHS.append(
            os.path.join(CHECKPOINT_DIR, f"checkpoints_rand-m{m}-n{n}/CarsVal/finetuned.pt")
        )
assert len(FINETUNED_PATHS) == 30


def build_args():
    args = argparse.Namespace()
    args.data_location = os.path.join(BASE_DIR, "dataset")
    args.finetuning_mode = "standard"
    args.model = "ViT-B-32"
    args.results_db = os.path.join(BASE_DIR, "checkpoints")
    args.save = os.path.join(args.results_db, args.finetuning_mode, args.model)
    args.batch_size = 128
    args.num_workers = 4
    args.openclip_cachedir = os.path.expanduser("~/openclip-cachedir/open_clip")
    args.cache_dir = None
    args.auto_aug = None
    args.train_dataset = None

    if torch.backends.mps.is_available():
        args.device = "mps"
    elif torch.cuda.is_available():
        args.device = "cuda"
    else:
        args.device = "cpu"
    return args


def load_state_dict_from_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
    return {k: v.cpu() for k, v in state_dict.items()}


_LOADER_ARGS = None


def _loader_args():
    global _LOADER_ARGS
    if _LOADER_ARGS is None:
        _LOADER_ARGS = argparse.Namespace(
            model="ViT-B-32",
            openclip_cachedir=os.path.expanduser("~/openclip-cachedir/open_clip"),
            cache_dir=None,
            auto_aug=None,
            train_dataset=None,
        )
    return _LOADER_ARGS


def load_image_encoder(checkpoint_path):
    """Reconstruct a fresh ImageEncoder, load the state dict from checkpoint.

    The monkey-patched CustomMultiheadAttention architecture gets rebuilt from
    scratch — older pickle-saved encoders may not have the right class structure.
    """
    state_dict = load_state_dict_from_ckpt(checkpoint_path)
    encoder = ImageEncoder(_loader_args(), keep_lang=False)
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()
    return encoder


class FreshNonLinearTaskVector(NonLinearTaskVector):
    """NonLinearTaskVector that rebuilds the encoder from scratch on apply.

    Matches the approach validated in the reference notebook.
    """

    def _load_checkpoint(self, checkpoint):
        return load_image_encoder(checkpoint)


def merge_subset(subset_paths, pretrained_state_dict):
    """Compute the sign-consensus merged task vector over subset_paths.

    Returns (vector_dict, consensus_rate). Sign consensus: a parameter is kept
    (averaged) iff all N task vectors share the same sign at that position.
    """
    N = len(subset_paths)
    task_vector_keys = []
    merged_vector = {}
    mask = {}

    for idx, finetuned_path in enumerate(subset_paths):
        ft_state_dict = load_state_dict_from_ckpt(finetuned_path)

        if idx == 0:
            for key in pretrained_state_dict:
                if pretrained_state_dict[key].dtype in (torch.int64, torch.uint8):
                    continue
                task_vector_keys.append(key)
                merged_vector[key] = torch.zeros_like(pretrained_state_dict[key])
                mask[key] = torch.zeros_like(pretrained_state_dict[key])

        for key in task_vector_keys:
            diff = ft_state_dict[key] - pretrained_state_dict[key]
            merged_vector[key] += diff
            mask[key] += torch.sign(diff)

        del ft_state_dict

    final_vector = {}
    total_params = 0
    consensus_params = 0
    for key in task_vector_keys:
        consistency_mask = torch.abs(mask[key]) == N
        final_vector[key] = torch.where(
            consistency_mask,
            merged_vector[key] / N,
            torch.zeros_like(merged_vector[key]),
        )
        total_params += final_vector[key].numel()
        consensus_params += int(consistency_mask.sum().item())

    consensus_rate = consensus_params / total_params if total_params > 0 else 0.0
    return final_vector, consensus_rate


def eval_at_alpha(task_vector, pretrained_path, args, alpha):
    # Use the *Val splits to match the baseline reproduction numbers
    # (CarsVal = 10% train split ≈ 814 samples; ImageNetVal = 10% of the local
    # val dir). Plain "Cars" / "ImageNet" would pull the full test sets and
    # land at different accuracies.
    args.eval_datasets = ["CarsVal"]
    args.control_dataset = "ImageNetVal"
    metrics = evaluate_task_vector_at_coef(
        -task_vector,
        pretrained_path,
        args,
        alpha,
    )
    return metrics["CarsVal:top1"], metrics["ImageNetVal:top1"]


def draw_subset(N, seed):
    if N == 30:
        return list(range(30))
    rng = random.Random(seed)
    return sorted(rng.sample(range(30), N))


def find_optimal(alpha_sweep):
    """Largest α with imagenet_acc >= threshold."""
    passing = [s for s in alpha_sweep if s["pass"]]
    if not passing:
        return None
    best = max(passing, key=lambda s: s["alpha"])
    return {
        "alpha": best["alpha"],
        "cars_acc": best["cars_acc"],
        "imagenet_acc": best["imagenet_acc"],
    }


def load_existing_results():
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT) as f:
            return json.load(f)
    return []


def save_results(records):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_path = JSON_OUT + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp_path, JSON_OUT)

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["N", "seed", "consensus_rate", "optimal_alpha", "cars_acc", "imagenet_acc"]
        )
        for r in records:
            opt = r["optimal"]
            if opt is None:
                writer.writerow([r["N"], r["seed"], f"{r['consensus_rate']:.6f}", "", "", ""])
            else:
                writer.writerow([
                    r["N"],
                    r["seed"],
                    f"{r['consensus_rate']:.6f}",
                    f"{opt['alpha']:.2f}",
                    f"{opt['cars_acc']:.6f}",
                    f"{opt['imagenet_acc']:.6f}",
                ])


def validate_n30_record(record):
    """Check a (N=30, seed=0) record against the published baseline. Returns (ok, msg)."""
    rate = record["consensus_rate"]
    if abs(rate - BASELINE_CONSENSUS) > TOL_CONSENSUS:
        return False, (
            f"consensus_rate {rate:.4f} differs from baseline {BASELINE_CONSENSUS:.4f} "
            f"by > {TOL_CONSENSUS:.4f}"
        )
    sweep = record["alpha_sweep"]
    a95 = next((s for s in sweep if abs(s["alpha"] - 0.95) < 1e-6), None)
    if a95 is None:
        return False, "no α=0.95 entry in alpha_sweep"
    if abs(a95["cars_acc"] - BASELINE_CARS_AT_095) > TOL_ACCURACY:
        return False, (
            f"Cars@α=0.95 = {a95['cars_acc']:.4f}, expected "
            f"{BASELINE_CARS_AT_095:.4f} ± {TOL_ACCURACY:.4f}"
        )
    if abs(a95["imagenet_acc"] - BASELINE_IMAGENET_AT_095) > TOL_ACCURACY:
        return False, (
            f"ImageNet@α=0.95 = {a95['imagenet_acc']:.4f}, expected "
            f"{BASELINE_IMAGENET_AT_095:.4f} ± {TOL_ACCURACY:.4f}"
        )
    return True, (
        f"consensus={rate:.4f}, Cars@0.95={a95['cars_acc']:.4f}, "
        f"ImageNet@0.95={a95['imagenet_acc']:.4f}"
    )


def print_summary(records):
    print()
    print("=" * 80)
    print("SUMMARY TABLE (means across seeds where α passed threshold)")
    print("=" * 80)
    header = f"{'N':<4} {'mean_consensus %':>18} {'mean_cars@opt %':>18} {'mean_imagenet@opt %':>22} {'n_passing':>10}"
    print(header)
    print("-" * len(header))
    for N in N_VALUES:
        all_rs = [r for r in records if r["N"] == N]
        rs = [r for r in all_rs if r["optimal"] is not None]
        if not all_rs:
            print(f"{N:<4} {'(missing)':>18}")
            continue
        if not rs:
            mean_consensus = np.mean([r["consensus_rate"] for r in all_rs])
            print(f"{N:<4} {mean_consensus*100:>18.2f} {'(no passing α)':>18}")
            continue
        mean_consensus = np.mean([r["consensus_rate"] for r in rs])
        mean_cars = np.mean([r["optimal"]["cars_acc"] for r in rs])
        mean_imagenet = np.mean([r["optimal"]["imagenet_acc"] for r in rs])
        print(
            f"{N:<4} {mean_consensus*100:>18.2f} {mean_cars*100:>18.2f} "
            f"{mean_imagenet*100:>22.2f} {len(rs):>10}"
        )


def run_one(N, seed, pretrained_state_dict, args):
    t0 = time.time()
    indices = draw_subset(N, seed)
    subset_paths = [FINETUNED_PATHS[i] for i in indices]
    print(f"[N={N} seed={seed}] indices={indices}")
    print(f"[N={N} seed={seed}] Merging {N} checkpoints ...")
    merged_vector, consensus_rate = merge_subset(subset_paths, pretrained_state_dict)
    print(f"[N={N} seed={seed}] consensus_rate={consensus_rate*100:.4f}%")

    task_vector = FreshNonLinearTaskVector(vector=merged_vector)

    alpha_sweep = []
    for alpha in ALPHAS:
        ta = time.time()
        print(f"[N={N} seed={seed}] Evaluating α={alpha:.2f} ...")
        cars_acc, imagenet_acc = eval_at_alpha(task_vector, PRETRAINED_PATH, args, alpha)
        passed = imagenet_acc >= IMAGENET_THRESHOLD
        alpha_sweep.append({
            "alpha": alpha,
            "cars_acc": cars_acc,
            "imagenet_acc": imagenet_acc,
            "pass": passed,
        })
        print(
            f"[N={N} seed={seed}]   α={alpha:.2f}  cars={cars_acc*100:.2f}%  "
            f"imagenet={imagenet_acc*100:.2f}%  pass={passed}  ({(time.time()-ta)/60:.1f} min)"
        )

    optimal = find_optimal(alpha_sweep)
    record = {
        "N": N,
        "seed": seed,
        "checkpoint_indices": indices,
        "consensus_rate": consensus_rate,
        "alpha_sweep": alpha_sweep,
        "optimal": optimal,
    }
    print(f"[N={N} seed={seed}] DONE ({(time.time()-t0)/60:.1f} min), optimal={optimal}")
    return record


def main():
    parser = argparse.ArgumentParser(description="NegMerge N-variation ablation")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the N=30 seed=0 baseline sanity check (use with caution).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated N values to run (e.g. '30' or '3,5'). Default: all.",
    )
    cli = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Pretrained: {PRETRAINED_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    print("Loading pretrained state dict ...")
    pretrained_state_dict = load_state_dict_from_ckpt(PRETRAINED_PATH)

    args = build_args()
    print(f"Eval device: {args.device}")

    records = load_existing_results()
    done_keys = {(r["N"], r["seed"]) for r in records}
    print(f"Existing records loaded: {len(records)} ({sorted(done_keys)})")

    ns_to_run = N_VALUES
    if cli.only is not None:
        requested = {int(x) for x in cli.only.split(",") if x.strip()}
        ns_to_run = [n for n in N_VALUES if n in requested]
        print(f"Running only N in {ns_to_run}")

    # (N=30, seed=0) first for validation, then remainder.
    pairs = []
    if 30 in ns_to_run:
        pairs.append((30, 0))
    for N in ns_to_run:
        seeds = SEEDS_N30 if N == 30 else SEEDS_SMALL_N
        for seed in seeds:
            if (N, seed) != (30, 0):
                pairs.append((N, seed))

    # Validation gate.
    n30_record = next((r for r in records if r["N"] == 30 and r["seed"] == 0), None)
    validated = False
    if n30_record is not None:
        if cli.skip_validation:
            print("Validation skipped per --skip-validation.")
            validated = True
        else:
            ok, msg = validate_n30_record(n30_record)
            if ok:
                print(f"[validation] PASS using cached N=30 seed=0 record: {msg}")
                validated = True
            else:
                print(f"[validation] FAIL on cached record: {msg}")
                print("Remove results/n_variation.json to recompute, or pass --skip-validation.")
                return 1

    for N, seed in pairs:
        if (N, seed) in done_keys:
            continue

        print()
        print("=" * 80)
        print(f"RUNNING N={N} seed={seed}")
        print("=" * 80)
        record = run_one(N, seed, pretrained_state_dict, args)
        records.append(record)
        done_keys.add((N, seed))
        save_results(records)

        if N == 30 and seed == 0 and not validated:
            ok, msg = validate_n30_record(record)
            if not ok:
                print()
                print("=" * 80)
                print(f"VALIDATION FAILED: {msg}")
                print("Stopping; not proceeding to smaller N. Inspect results/n_variation.json.")
                print("=" * 80)
                return 1
            print()
            print("=" * 80)
            print(f"VALIDATION PASSED: {msg}")
            print("=" * 80)
            validated = True

    print_summary(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
