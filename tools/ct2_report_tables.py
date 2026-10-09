"""Markdown tables for the round-2 report from the final kernel's result files (pure text processing; reads results only, no data).
usage: python tools/ct2_report_tables.py <dir with ct_final_ct_round2.json, ct_headline_ct_round2.json, ct_compare_ref_ct_round2.json, ct_edge_ct_round2.json>"""
import json, os, sys

d = sys.argv[1]
J = lambda n: json.load(open(os.path.join(d, n))) if os.path.exists(os.path.join(d, n)) else None
fin, head, cmp_, edge = J("ct_final_ct_round2.json"), J("ct_headline_ct_round2.json"), J("ct_compare_ref_ct_round2.json"), J("ct_edge_ct_round2.json")
if fin:
    print("| seed | test RMSE [V] | train RMSE [V] | 95 % block-bootstrap interval of this seed's RMSE | fit s | a | b4 | w3 |")
    print("|---|---|---|---|---|---|---|---|")
    for s in sorted(fin["seeds"], key=int):
        r = fin["seeds"][s]
        th = r["theta"]
        print(f"| {s} | {r['test_rmse']:.4f} | {r['train_rmse']:.5f} | {r['ci95_block_bootstrap'][0]:.4f} .. {r['ci95_block_bootstrap'][1]:.4f} | {r['fit_secs']:.0f} | {th['a']:.3f} | {th['b4']:.3f} | {th.get('w3', 0):.3f} |")
    print("summary:", json.dumps(fin["summary"]))
if head:
    print("\nheadline: mean %.4f  sd over seeds %.4f  min %.4f  max %.4f  95%% block-bootstrap interval of the mean [%.4f, %.4f]  P(bootstrap mean < 0.221) = %.4f  n_samples %d" % (
        head["mean_rmse"], head["sd_over_seeds"] or 0, head["min_rmse"], head["max_rmse"], head["mean_ci95_block_bootstrap"][0], head["mean_ci95_block_bootstrap"][1],
        head["prob_bootstrap_mean_below_target"], head["n_samples_scored"]))
    print("max |logged - recomputed| =", head.get("max_abs_diff_vs_logged_rmse"))
if cmp_:
    print("\ncompare_ref summary:", json.dumps(cmp_["summary"]))
    seeds = sorted(cmp_["pairs"], key=int)
    refs = sorted(cmp_["pairs"][seeds[0]]["vs_ref"], key=lambda r: cmp_["pairs"][seeds[0]]["vs_ref"][r]["ref_rmse"])
    print("\n| reference seed | its RMSE | ours (seed %s) minus reference | 95 %% paired interval | our 10 seeds: diff range | upper bound < 0 for how many of our seeds |" % seeds[0])
    print("|---|---|---|---|---|---|")
    for r in refs:
        v = cmp_["pairs"][seeds[0]]["vs_ref"][r]
        diffs = [cmp_["pairs"][s]["vs_ref"][r]["diff"] for s in seeds]
        nb = sum(cmp_["pairs"][s]["vs_ref"][r]["beats"] for s in seeds)
        lab = r + (" (notebook default)" if r == "42" else "")
        print(f"| {lab} | {v['ref_rmse']:.4f} | {v['diff']:+.4f} | {v['lo']:+.4f} .. {v['hi']:+.4f} | {min(diffs):+.4f} .. {max(diffs):+.4f} | {nb} of {len(seeds)} |")
if edge:
    print("\nedge: structure", edge["structure"], "numbers", edge["n_numbers_incl_x0"], "free", edge["n_params_free"], "raw payload bytes", edge["raw_payload_bytes"], "npy containers", edge["bytes_npy_containers"])
    for k, v in edge["quantisation_train_record"].items():
        print(f"  {k}: train RMSE {v['train_rmse']:.5f}  delta {v['delta_vs_fp64']:+.2e}  max traj dev {v['max_abs_dev_of_trajectory_vs_fp64']:.2e}  max rel param err {v['max_relative_parameter_error']:.2e}")
    print("  c99:", json.dumps(edge["c99"]))
    print("  python float ns/step:", edge.get("python_float_ns_per_step"))
