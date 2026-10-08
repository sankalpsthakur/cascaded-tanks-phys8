"""Headline aggregation for a frozen Cascaded Tanks run.

Analysis of the ALREADY LOGGED test predictions (runs/ct_final_<name>_seed*_ypred.npy, written by ct/final.py); it scores
no new model on the test record, it only re-reads the test labels to form residuals of those saved predictions
(the same thing ct/compare_ref.py does).

Headline (freeze `claim_rule`): mean over seeds of the official test RMSE (n_init = 50), never the best seed.
Uncertainty: moving-block bootstrap (block 25, 4000 draws, seed 0) of the test residuals; the SAME resampled time indices
are applied to all seeds, so the interval is for the mean-over-seeds statistic, not the seed spread.

usage: python tools/ct_aggregate.py <freeze-name>      (e.g. ct_phys8)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BLOCK, NBOOT, N_INIT, TARGET = 25, 4000, 50, 0.221


def block_indices(n, rng, block=BLOCK):
    nb = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    return (rng.choice(starts, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]


def mean_seed_bootstrap(E, n_boot=NBOOT, block=BLOCK, seed=0):
    """E: (S, n) residuals (one row per seed). Returns bootstrap draws of (mean over seeds of RMSE) and the per-seed RMSE draws."""
    S, n = E.shape
    sq = E ** 2
    rng = np.random.default_rng(seed)
    per = np.empty((n_boot, S))
    for i in range(n_boot):
        idx = block_indices(n, rng, block)
        per[i] = np.sqrt(sq[:, idx].mean(axis=1))
    return per.mean(axis=1), per


def summarise(E, seeds, logged=None):
    rm = np.sqrt((E ** 2).mean(axis=1))
    draws, per = mean_seed_bootstrap(E)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    out = dict(
        n_seeds=int(E.shape[0]), n_samples_scored=int(E.shape[1]),
        per_seed_rmse={str(s): float(r) for s, r in zip(seeds, rm)},
        mean_rmse=float(rm.mean()), sd_over_seeds=float(rm.std(ddof=1)) if len(rm) > 1 else None,
        min_rmse=float(rm.min()), max_rmse=float(rm.max()), median_rmse=float(np.median(rm)),
        mean_ci95_block_bootstrap=[float(lo), float(hi)],
        bootstrap=dict(block=BLOCK, n_boot=NBOOT, seed=0, resampled_indices_shared_across_seeds=True),
        prob_bootstrap_mean_below_target=float((draws < TARGET).mean()), target=TARGET,
        max_per_seed_ci95_hi=float(np.quantile(per, 0.975, axis=0).max()),
        min_per_seed_ci95_lo=float(np.quantile(per, 0.025, axis=0).min()),
    )
    if logged is not None:
        out["max_abs_diff_vs_logged_rmse"] = float(max(abs(rm[i] - logged[s]) for i, s in enumerate(seeds)))
    return out


def main(name):
    from ct import protocol as P  # noqa: E402

    res = json.load(open(os.path.join(ROOT, "results", f"ct_final_{name}.json")))
    _, test = P._load_both()
    yt = np.asarray(test.y, float).ravel()
    seeds = sorted(res["seeds"], key=int)
    E = np.stack([np.load(os.path.join(ROOT, "runs", f"ct_final_{name}_seed{s}_ypred.npy"))[N_INIT:] - yt[N_INIT:] for s in seeds])
    logged = {s: res["seeds"][s]["test_rmse"] for s in seeds}
    out = summarise(E, seeds, logged)
    out["name"] = name
    out["frozen_run_summary"] = res.get("summary")
    cmp_path = os.path.join(ROOT, "results", f"ct_compare_ref_{name}.json")
    if os.path.exists(cmp_path):
        cmp = json.load(open(cmp_path))
        out["compare_ref_summary"] = cmp["summary"]
        beats = bool(cmp["summary"]["beats_0221_entry_every_seed"])
    else:
        beats = None
    out["claim"] = dict(
        rule="freeze claim_rule: 'best published result' wording only if the paired 95% upper bound vs the 0.221-entry code (seed 42) is < 0 for EVERY one of our seeds",
        beats_0221_entry_every_seed=beats,
        mean_below_0221=bool(out["mean_rmse"] < TARGET),
        wording=("best published result on Cascaded Tanks, official split, n_init=50 free-run simulation RMSE, self-evaluated, as of 2026-10-07"
                 if beats else
                 "does NOT meet the pre-registered rule for a 'best published result' claim; report the numbers as they are"),
    )
    json.dump(out, open(os.path.join(ROOT, "results", f"ct_headline_{name}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_seeds", "mean_rmse", "sd_over_seeds", "min_rmse", "max_rmse", "mean_ci95_block_bootstrap",
                                           "prob_bootstrap_mean_below_target", "max_abs_diff_vs_logged_rmse")}, indent=1))
    print("claim:", json.dumps(out["claim"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
