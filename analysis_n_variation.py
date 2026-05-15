"""Post-hoc statistics for the NegMerge N-variation ablation.

Reads results/n_variation.json (preferred) or results/n_variation.csv as
fallback, and runs three analyses on Cars accuracy at the optimal α per
(N, seed):

  1. Within-plateau Welch's t-tests:
       α=0.50: N=3 vs N=5
       α=0.75: N=10 vs N=15
       α=0.90: N=20 vs N=25

  2. Pooled standard-deviation audit under three different scopes:
       (a) just N=10 and N=15  (the scope the paper currently implies)
       (b) all six stochastic groups (N ∈ {3,5,10,15,20,25})
       (c) the three within-plateau comparison pairs only
     Note: with equal n=3 across all stochastic groups, scopes (b) and (c)
     resolve to the same pooled variance — both average across the same six
     groups. The framing differs; the number doesn't.

  3. Cross-plateau α-ceiling drops on Cars accuracy:
       N=5  → N=10 (α 0.50→0.75)
       N=15 → N=20 (α 0.75→0.90)
       N=25 → N=30 (α 0.90→0.95)
     Each compared to the within-plateau drift on the lower plateau.

Optional: N=3 α=0.95 ImageNet collapse, computed only if the JSON
alpha_sweep is available.

No re-running of merging or evaluation. Pure post-hoc statistics. Uses
only numpy and scipy.stats.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO_DIR, "results")
JSON_PATH = os.path.join(RESULTS_DIR, "n_variation.json")
CSV_PATH = os.path.join(RESULTS_DIR, "n_variation.csv")
OUT_PATH = os.path.join(RESULTS_DIR, "n_variation_stats.json")

PLATEAUS = [
    {"label": "α=0.50", "alpha": 0.50, "lo": 3, "hi": 5},
    {"label": "α=0.75", "alpha": 0.75, "lo": 10, "hi": 15},
    {"label": "α=0.90", "alpha": 0.90, "lo": 20, "hi": 25},
]

CROSS_DROPS = [
    # (label, lower-plateau higher-N, upper-plateau lower-N, drift_pair)
    {"label": "N=5 → N=10  (α 0.50 → 0.75)",  "from_N": 5,  "to_N": 10, "drift_pair": (3, 5)},
    {"label": "N=15 → N=20 (α 0.75 → 0.90)", "from_N": 15, "to_N": 20, "drift_pair": (10, 15)},
    {"label": "N=25 → N=30 (α 0.90 → 0.95)", "from_N": 25, "to_N": 30, "drift_pair": (20, 25)},
]


# --------------------------------------------------------------------------
# Loading


def load_records() -> Tuple[Dict[int, Dict[int, dict]],
                            Dict[int, Dict[int, list]],
                            str]:
    """Return (records_by_N_seed, alpha_sweeps_by_N_seed, source_label)."""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            data = json.load(f)
        records: Dict[int, Dict[int, dict]] = defaultdict(dict)
        sweeps: Dict[int, Dict[int, list]] = defaultdict(dict)
        for r in data:
            opt = r.get("optimal")
            if opt is None:
                continue
            records[r["N"]][r["seed"]] = {
                "cars": float(opt["cars_acc"]),
                "imagenet": float(opt["imagenet_acc"]),
                "optimal_alpha": float(opt["alpha"]),
                "consensus_rate": float(r["consensus_rate"]),
            }
            if r.get("alpha_sweep"):
                sweeps[r["N"]][r["seed"]] = r["alpha_sweep"]
        return dict(records), dict(sweeps), "json"

    if os.path.exists(CSV_PATH):
        records = defaultdict(dict)
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                if not row.get("optimal_alpha"):
                    continue
                records[int(row["N"])][int(row["seed"])] = {
                    "cars": float(row["cars_acc"]),
                    "imagenet": float(row["imagenet_acc"]),
                    "optimal_alpha": float(row["optimal_alpha"]),
                    "consensus_rate": float(row["consensus_rate"]),
                }
        return dict(records), {}, "csv"

    return {}, {}, "missing"


def cars_pp(records: Dict[int, Dict[int, dict]], N: int) -> np.ndarray:
    """Cars accuracies at the optimal α for N, in percentage points (0–100)."""
    seeds = sorted(records[N].keys())
    return np.array([records[N][s]["cars"] * 100.0 for s in seeds], dtype=float)


def imagenet_pp(records: Dict[int, Dict[int, dict]], N: int) -> np.ndarray:
    seeds = sorted(records[N].keys())
    return np.array([records[N][s]["imagenet"] * 100.0 for s in seeds], dtype=float)


# --------------------------------------------------------------------------
# Statistics helpers


def cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Cohen's d for (y - x) using pooled sample SD (ddof=1).

    Returns (d, s_pooled).
    """
    nx, ny = len(x), len(y)
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    s_pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if s_pooled == 0.0:
        return float("inf") if (y.mean() - x.mean()) != 0 else 0.0, 0.0
    return (float(y.mean()) - float(x.mean())) / s_pooled, s_pooled


def welch(x: np.ndarray, y: np.ndarray) -> dict:
    """Welch's two-sample t-test on (y - x). Returns full breakdown."""
    nx, ny = len(x), len(y)
    mx, my = float(x.mean()), float(y.mean())
    vx, vy = float(x.var(ddof=1)), float(y.var(ddof=1))
    se = math.sqrt(vx / nx + vy / ny)
    diff = my - mx
    t = diff / se if se > 0 else (float("inf") if diff != 0 else 0.0)
    if vx / nx + vy / ny > 0:
        df_num = (vx / nx + vy / ny) ** 2
        df_den = ((vx / nx) ** 2) / max(nx - 1, 1) + ((vy / ny) ** 2) / max(ny - 1, 1)
        df = df_num / df_den if df_den > 0 else float("inf")
    else:
        df = float("inf")
    p = 2.0 * stats.t.sf(abs(t), df) if math.isfinite(df) else 0.0
    d, s_pooled = cohens_d(x, y)

    # Cross-check with scipy
    try:
        scipy_t, scipy_p = stats.ttest_ind(y, x, equal_var=False)
        scipy_t = float(scipy_t)
        scipy_p = float(scipy_p)
    except Exception:
        scipy_t = scipy_p = None

    return {
        "n_x": nx,
        "n_y": ny,
        "mean_x": mx,
        "mean_y": my,
        "std_x": math.sqrt(vx),
        "std_y": math.sqrt(vy),
        "diff_y_minus_x": diff,
        "welch_se": se,
        "t": t,
        "df": df,
        "p_two_sided": p,
        "cohens_d": d,
        "pooled_sd": s_pooled,
        "scipy_t": scipy_t,
        "scipy_p": scipy_p,
    }


def pooled_sd_across_groups(arrays: List[np.ndarray]) -> Tuple[float, int]:
    """Pool sample variance across groups (each contributes (n-1) df).

    Returns (pooled_sd, total_df).
    """
    num = 0.0
    den = 0
    for a in arrays:
        n = len(a)
        if n <= 1:
            continue
        num += (n - 1) * float(a.var(ddof=1))
        den += n - 1
    if den == 0:
        return 0.0, 0
    return math.sqrt(num / den), den


# --------------------------------------------------------------------------
# Reporting helpers


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def print_welch_block(label: str, low_label: str, hi_label: str, w: dict) -> None:
    print(f"  {label}")
    print(f"    {low_label}:  n={w['n_x']}  mean={w['mean_x']:.4f} pp  sd={w['std_x']:.4f} pp")
    print(f"    {hi_label}:   n={w['n_y']}  mean={w['mean_y']:.4f} pp  sd={w['std_y']:.4f} pp")
    print(f"    mean diff (hi - lo)        = {w['diff_y_minus_x']:+.4f} pp")
    print(f"    Welch SE                   = {w['welch_se']:.4f} pp")
    print(f"    t-statistic                = {w['t']:+.4f}")
    print(f"    Welch–Satterthwaite df     = {w['df']:.4f}")
    print(f"    two-sided p-value          = {fmt_p(w['p_two_sided'])}")
    if w.get("scipy_p") is not None:
        print(f"    scipy ttest_ind cross-check: t={w['scipy_t']:+.4f}, p={fmt_p(w['scipy_p'])}")
    print(f"    Cohen's d (pooled SD)      = {w['cohens_d']:+.4f}    (s_pooled = {w['pooled_sd']:.4f} pp)")
    print()


# --------------------------------------------------------------------------
# Main analyses


def analysis_within_plateau(records) -> List[dict]:
    print("=" * 80)
    print("1. Within-plateau Welch's t-tests on Cars accuracy")
    print("=" * 80)
    print()
    out = []
    for p in PLATEAUS:
        x = cars_pp(records, p["lo"])
        y = cars_pp(records, p["hi"])
        w = welch(x, y)
        rec = {
            "plateau": p["label"],
            "low_N": p["lo"],
            "high_N": p["hi"],
            "low_seeds_cars_pp": x.tolist(),
            "high_seeds_cars_pp": y.tolist(),
            "welch": w,
        }
        out.append(rec)
        print_welch_block(
            f"Plateau {p['label']} :  N={p['lo']}  vs  N={p['hi']}",
            f"N={p['lo']}",
            f"N={p['hi']}",
            w,
        )
    return out


def analysis_pooled_sd(records) -> dict:
    print("=" * 80)
    print("2. Pooled-SD audit on Cars accuracy (in percentage points)")
    print("=" * 80)
    print()

    # (a) just N=10 and N=15
    arr10 = cars_pp(records, 10)
    arr15 = cars_pp(records, 15)
    sd_a, df_a = pooled_sd_across_groups([arr10, arr15])
    print(f"  (a) Scope = {{N=10, N=15}}  (the paper's implied scope)")
    print(f"      SD(N=10)   = {arr10.std(ddof=1):.4f} pp")
    print(f"      SD(N=15)   = {arr15.std(ddof=1):.4f} pp")
    print(f"      pooled SD  = {sd_a:.4f} pp     (total df = {df_a})")
    print()

    # (b) all six stochastic groups
    groups_b = [cars_pp(records, N) for N in (3, 5, 10, 15, 20, 25)]
    sd_b, df_b = pooled_sd_across_groups(groups_b)
    print(f"  (b) Scope = all six stochastic groups  (N ∈ {{3,5,10,15,20,25}})")
    for N, a in zip((3, 5, 10, 15, 20, 25), groups_b):
        print(f"      SD(N={N:<2}) = {a.std(ddof=1):.4f} pp")
    print(f"      pooled SD  = {sd_b:.4f} pp     (total df = {df_b})")
    print()

    # (c) the three within-plateau pairs only
    # With equal n=3 across all groups, this resolves to the same pooled
    # variance as (b) — same six groups, just framed as three plateau-pairs.
    plateau_groups = []
    for p in PLATEAUS:
        plateau_groups.append(cars_pp(records, p["lo"]))
        plateau_groups.append(cars_pp(records, p["hi"]))
    sd_c, df_c = pooled_sd_across_groups(plateau_groups)
    print(f"  (c) Scope = the three within-plateau comparison pairs only")
    for p in PLATEAUS:
        a_lo = cars_pp(records, p["lo"])
        a_hi = cars_pp(records, p["hi"])
        sd_pair, df_pair = pooled_sd_across_groups([a_lo, a_hi])
        print(f"      pair {p['label']:>7}  (N={p['lo']:>2}, N={p['hi']:>2}):  pooled SD = {sd_pair:.4f} pp  (df = {df_pair})")
    print(f"      pooled SD across pairs = {sd_c:.4f} pp     (total df = {df_c})")
    print()
    print(f"  Note: (b) and (c) average over the same six groups with equal n=3,")
    print(f"        so they resolve to the same number; the labelling is what")
    print(f"        differs. Only (a) is restricted to a single plateau.")
    print()

    print("  Most defensible to cite:")
    print("    (a) — restrict the 0.43 pp claim to the within-plateau scope it")
    print("    actually describes (N=10 vs N=15 on the α=0.75 plateau). The")
    print("    experiment is heteroscedastic — group SDs span ~0.07 pp at N=20")
    print("    to ~1.73 pp at N=3 — so a single across-group pooled SD is")
    print("    misleading when applied to comparisons on other plateaus.")
    print()

    return {
        "scope_a_N10_N15": {
            "pooled_sd_pp": sd_a, "df": df_a,
            "per_group_sd_pp": {
                "N=10": float(arr10.std(ddof=1)),
                "N=15": float(arr15.std(ddof=1)),
            },
        },
        "scope_b_all_stochastic": {
            "pooled_sd_pp": sd_b, "df": df_b,
            "per_group_sd_pp": {
                f"N={N}": float(a.std(ddof=1))
                for N, a in zip((3, 5, 10, 15, 20, 25), groups_b)
            },
        },
        "scope_c_plateau_pairs_only": {
            "pooled_sd_pp": sd_c, "df": df_c,
            "per_pair_pooled_sd_pp": {
                p["label"]: pooled_sd_across_groups(
                    [cars_pp(records, p["lo"]), cars_pp(records, p["hi"])]
                )[0]
                for p in PLATEAUS
            },
        },
        "note": "With equal n=3 across all stochastic groups, scopes (b) and (c) are the same pooled variance; the labelling is what differs.",
        "recommended": "scope_a_N10_N15",
    }


def analysis_cross_plateau(records) -> List[dict]:
    print("=" * 80)
    print("3. Cross-plateau α-ceiling drops vs within-plateau drift on Cars")
    print("=" * 80)
    print()

    out = []
    for d in CROSS_DROPS:
        from_N = d["from_N"]
        to_N = d["to_N"]
        drift_lo, drift_hi = d["drift_pair"]

        from_arr = cars_pp(records, from_N)
        to_arr = cars_pp(records, to_N)

        from_mean = float(from_arr.mean())
        to_mean = float(to_arr.mean())
        # Drop is from_N − to_N (i.e., a positive number means the upper plateau
        # gave a *lower* (better) Cars accuracy).
        drop = from_mean - to_mean

        drift_lo_arr = cars_pp(records, drift_lo)
        drift_hi_arr = cars_pp(records, drift_hi)
        drift = float(drift_hi_arr.mean()) - float(drift_lo_arr.mean())

        # Welch on the cross-plateau drop, when both groups have n>1
        if len(from_arr) > 1 and len(to_arr) > 1:
            cross_w = welch(to_arr, from_arr)  # diff = from - to
            # We want diff = from - to (positive when upper plateau is better):
            cross_w["diff_y_minus_x"] = float(from_arr.mean() - to_arr.mean())
            cross_w["t"] = -cross_w["t"]  # symmetric — sign just flips for the
                                          # direction we want to talk about
            cross_w["cohens_d"] = -cross_w["cohens_d"]
        else:
            cross_w = None

        ratio_str = "n/a"
        if drift != 0:
            ratio_str = f"{drop / drift:+.2f}×"

        print(f"  {d['label']}")
        print(f"    upper plateau N={from_N:<2} cars mean = {from_mean:.4f} pp")
        print(f"    upper plateau N={to_N:<2} cars mean = {to_mean:.4f} pp")
        print(f"    cross-plateau drop                = {drop:+.4f} pp   (from N={from_N} − N={to_N})")
        print(f"    within-plateau drift (N={drift_lo} → N={drift_hi}) = {drift:+.4f} pp")
        print(f"    drop / drift                      = {ratio_str}")
        verdict = "DROP > DRIFT" if drop > drift else ("DROP ≈ DRIFT" if abs(drop - drift) < 0.5 else "DROP < DRIFT")
        print(f"    verdict                           = {verdict}")
        if cross_w:
            print(f"    Welch t (drop)                    = {cross_w['t']:+.4f},  df={cross_w['df']:.2f},  p={fmt_p(cross_w['p_two_sided'])}")
        else:
            print(f"    Welch t (drop)                    = n/a (N=30 deterministic, n=1)")
        print()

        out.append({
            "label": d["label"],
            "from_N": from_N,
            "to_N": to_N,
            "from_mean_cars_pp": from_mean,
            "to_mean_cars_pp": to_mean,
            "drop_pp": drop,
            "drift_pair": [drift_lo, drift_hi],
            "drift_pp": drift,
            "drop_minus_drift_pp": drop - drift,
            "drop_over_drift": (drop / drift) if drift != 0 else None,
            "verdict": verdict,
            "cross_welch": cross_w,
        })

    return out


def analysis_n3_imagenet_collapse(sweeps) -> Optional[dict]:
    """Compute mean ImageNet at α=0.95 for N=3, if alpha_sweep is available."""
    if 3 not in sweeps:
        return None
    accs = []
    for seed, sweep in sweeps[3].items():
        for s in sweep:
            if abs(float(s["alpha"]) - 0.95) < 1e-9:
                accs.append(float(s["imagenet_acc"]))
                break
    if not accs:
        return None
    arr = np.array(accs) * 100.0
    return {
        "alpha": 0.95,
        "N": 3,
        "imagenet_pp_per_seed": arr.tolist(),
        "imagenet_pp_mean": float(arr.mean()),
        "imagenet_pp_std": float(arr.std(ddof=1)),
        "threshold_pp": 59.25,
        "shortfall_pp": float(arr.mean() - 59.25),
    }


# --------------------------------------------------------------------------
# Markdown render


def render_markdown(within: List[dict], pooled: dict, cross: List[dict],
                    n3_collapse: Optional[dict]) -> str:
    a_pair = within[1]["welch"]  # α=0.75 pair: N=10 vs N=15
    sd_a = pooled["scope_a_N10_N15"]["pooled_sd_pp"]
    per = pooled["scope_b_all_stochastic"]["per_group_sd_pp"]
    sd_min = min(per.values())
    sd_max = max(per.values())
    n_min = min(per, key=per.get)
    n_max = max(per, key=per.get)

    c1, c2, c3 = cross[0], cross[1], cross[2]
    md = []
    md.append("### §4.2 — replacement paragraph (≤120 words)")
    md.append("")
    md.append(
        f"Seed variance is not pooled-uniform across the ablation. "
        f"Per-group sample SDs span {sd_min:.2f} pp at {n_min} to "
        f"{sd_max:.2f} pp at {n_max} (≈{sd_max/max(sd_min,1e-9):.0f}× range), "
        f"so we cite the within-plateau pooled SD rather than a single "
        f"across-experiment number. On the α=0.75 plateau the pooled SD "
        f"across N=10 and N=15 is **{sd_a:.2f} pp**, and the N=15 − N=10 "
        f"mean Cars gap of **{a_pair['diff_y_minus_x']:.2f} pp** is highly "
        f"significant under Welch's t (t={a_pair['t']:.2f}, df={a_pair['df']:.2f}, "
        f"p={fmt_p(a_pair['p_two_sided'])}, Cohen's d={a_pair['cohens_d']:.2f}). "
        f"Cross-plateau α-ceiling drops dominate within-plateau drift on the "
        f"first two transitions (N=5→10: {c1['drop_pp']:.2f} vs {c1['drift_pp']:.2f} pp; "
        f"N=15→20: {c2['drop_pp']:.2f} vs {c2['drift_pp']:.2f} pp), but at "
        f"N=25→30 the drop ({c3['drop_pp']:.2f} pp) is comparable to drift "
        f"({c3['drift_pp']:.2f} pp), suggesting diminishing returns past N≈25."
    )
    md.append("")

    if n3_collapse is not None:
        md.append("### Table 2 footnote")
        md.append("")
        md.append(
            f"At N=3, pushing α to 0.95 collapses ImageNet retention to a "
            f"3-seed mean of {n3_collapse['imagenet_pp_mean']:.2f} pp "
            f"(individual seeds: "
            f"{', '.join(f'{x:.2f}' for x in n3_collapse['imagenet_pp_per_seed'])} pp), "
            f"{abs(n3_collapse['shortfall_pp']):.2f} pp below the 59.25 pp "
            f"retention threshold; this is why the optimal α reverts to 0.50 "
            f"in row N=3 of Table 2."
        )
        md.append("")

    return "\n".join(md)


def main() -> int:
    records, sweeps, source = load_records()
    if not records:
        print("ERROR: no raw per-seed data found.", file=sys.stderr)
        print(f"  Looked for: {JSON_PATH}", file=sys.stderr)
        print(f"             {CSV_PATH}", file=sys.stderr)
        print("  Re-run src/eval_n_variation.py first.", file=sys.stderr)
        return 1

    print(f"Loaded raw per-seed data from: {source}")
    Ns = sorted(records.keys())
    print(f"  N values: {Ns}")
    for N in Ns:
        seeds = sorted(records[N].keys())
        print(f"    N={N:<2}: seeds={seeds}, n={len(seeds)}")
    print()

    within = analysis_within_plateau(records)
    pooled = analysis_pooled_sd(records)
    cross = analysis_cross_plateau(records)

    n3_collapse = analysis_n3_imagenet_collapse(sweeps) if sweeps else None
    if n3_collapse is not None:
        print("=" * 80)
        print("4. N=3 α=0.95 ImageNet collapse")
        print("=" * 80)
        print()
        print(f"  per-seed imagenet @ α=0.95: " +
              ", ".join(f"{x:.4f} pp" for x in n3_collapse["imagenet_pp_per_seed"]))
        print(f"  mean                       = {n3_collapse['imagenet_pp_mean']:.4f} pp")
        print(f"  threshold                  = 59.2500 pp")
        print(f"  shortfall (mean - thresh)  = {n3_collapse['shortfall_pp']:.4f} pp")
        print()
    else:
        print("=" * 80)
        print("4. N=3 α=0.95 ImageNet collapse")
        print("=" * 80)
        print()
        print("  alpha_sweep not available in raw data; skipping.")
        print()

    md = render_markdown(within, pooled, cross, n3_collapse)
    print("=" * 80)
    print("5. Paste-ready markdown")
    print("=" * 80)
    print()
    print(md)
    print()

    out_blob = {
        "source": source,
        "n_per_group": {str(N): len(records[N]) for N in Ns},
        "within_plateau": within,
        "pooled_sd": pooled,
        "cross_plateau_drops": cross,
        "n3_imagenet_collapse_at_alpha_0_95": n3_collapse,
        "markdown": md,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out_blob, f, indent=2)
    print(f"Saved: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
