"""Turn the selection kernel's selection.json into results/FREEZE_ct_round2.json, mechanically (no hand picking).

usage: python ct2/make_freeze.py <selection.json> <kernel ref + version> <out.json> [--date 2026-10-09]
Reads files only. The claim rule is copied verbatim from results/FREEZE_ct_phys8.json (only the 'as of' date in the wording changes).
"""
import argparse, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TERMS = {"tp": "first-order pump/inflow lag", "w3": "level-dependent lower-outlet coefficient", "to": "delayed return of the upper-tank overflow",
         "qov": "flow-dependent overflow routing exponent", "c4": "pump curvature", "h0": "upper-tank outlet height", "d1": "input FIR taps",
         "d2": "input FIR taps", "cA1": "upper-tank capacity slope"}

DESIGN_INFO = ("Round 2 is NOT an independent test of the test record. The candidate families (pump lag, level-dependent lower outlet, delayed overflow return, "
               "flow-dependent routing, upper outlet height, input FIR) were chosen after the round-1 error analysis of the TEST residuals "
               "(REPORT-CT.md section 4: 86 % of the squared test error on the fast rise to the second large peak and its under-shoot; test minimum 2.16 V below "
               "the estimation minimum 2.91 V; round-1 test RMSE 0.2719 V). The selection itself used the estimation record only. No round-2 code reads test values before "
               "this freeze, and the saved round-1 test predictions were not opened. nonlinearbenchmark.org's reporting rules forbid using test data to tune structure; "
               "a board resubmission of a round-2 result would need the lead to weigh that rule.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("selection"); ap.add_argument("kernel"); ap.add_argument("out")
    ap.add_argument("--date", default="2026-10-09")
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    if "final" not in sel:
        sys.exit("selection.json has no 'final' block: the selection did not finish (" + str(sel.get("abort")) + ")")
    r1 = json.load(open(os.path.join(ROOT, "results", "FREEZE_ct_phys8.json")))
    claim = json.loads(json.dumps(r1["claim_rule"]))
    claim["wording"] = claim["wording"].replace("as of 2026-10-07", f"as of {a.date}")
    fin = sel["final"]
    vs = fin["variant_spec"]
    ident = fin["identical_to_ct_phys8"]
    free = fin["free"]
    extras = [TERMS[n] for n in free if n in TERMS]
    cand_label = sel["stage1"]["rows"][fin["structure"]]["label"]

    def compact(rows):
        keep = ("cand", "variant", "label", "n_numbers", "folds", "cv", "se", "level_rmse", "complete", "secs", "error")
        return {k: {kk: v[kk] for kk in keep if kk in v} for k, v in rows.items()}

    cfg = {
        "name": "ct_round2",
        "round": 2,
        "description": ("Round-2 grey-box two-tank Torricelli ODE selected by the pre-registered rule of PREREGISTRATION-CT-ROUND2.md: structure %s (%s), fit %s (%s). "
                        "Base of ct_phys8 (bottle-shaped lower tank, signed pump offset, overflow clamp with routed surplus, sensor offset)%s. %d free physical parameters; the 2 initial states "
                        "are fitted per record from the first 50 samples only." % (fin["structure"], cand_label, fin["variant"], vs["label"],
                                                                                   (" plus: " + ", ".join(sorted(set(extras)))) if extras else "", len(free))),
        "preregistration": "PREREGISTRATION-CT-ROUND2.md",
        "structure": {"id": fin["structure"], "name": f"{fin['structure']} {cand_label}", "free": free, "clamp_x2": True, "nsub": 4, "n_numbers": len(free) + 2},
        "fit": {"variant": fin["variant"], "label": vs["label"], "loss": vs["loss"], "huber_delta": vs.get("huber_delta"), "ms_K": vs.get("ms_K", 0),
                "ms_lam": vs.get("ms_lam"), "n_starts": vs["final_starts"], "maxiter": fin["final_maxiter"],
                "optimiser": "scipy L-BFGS-B on JAX float64 gradients, objective as stated, on the full estimation record; best start by objective"},
        "initial_state": "per-record: x0 fitted on the first 50 samples (the official state-initialisation window) with all other parameters frozen; a pump-lag state starts at the previous input (steady state), a delayed-overflow buffer starts empty (no extra fitted numbers)",
        "selection": {"kernel": a.kernel, "reproduction_gate": sel.get("reproduction_gate"), "stage1": compact(sel["stage1"]["rows"]), "rule1": sel["stage1"]["rule"],
                      "stage2_structure": sel["stage2"]["structure"], "stage2": compact(sel["stage2"]["rows"]), "rule2": sel["stage2"]["rule"],
                      "method": "see PREREGISTRATION-CT-ROUND2.md section 4; 4 contiguous masked blocks of 256 on the ESTIMATION record, level-extrapolation check (lowest 12 % masked), smallest structure within one standard error of the best pooled CV RMSE, ties by the extrapolation check"},
        "dry_run_train_only": {k: fin["dry_run"][k] for k in ("train_rmse", "objective", "theta", "x0", "secs")},
        "identical_to_ct_phys8": bool(ident.get("identical")),
        "identity_check": ident,
        "seeds": list(range(10)),
        "protocol": "official nonlinear_benchmarks metric RMSE(test, y_model, n_init=50), free-run simulation, test record scored once per seed via ct.protocol.test_eval (ct2/final2.py)",
        "claim_rule": claim,
        "claim_rule_note": "copied verbatim from results/FREEZE_ct_phys8.json; only the 'as of' date in the wording is the date of this run. A result at or above 0.221 V is reported as-is.",
        "test_record_information_used_in_design": DESIGN_INFO,
        "test_record_looked_at_before_freeze": ("Everything in results/FREEZE_ct_phys8.json 'test_record_looked_at_before_freeze' still applies, plus the round-1 scoring of ct_phys8 (TEST-LOG.md rows 15-24) and its error analysis. "
                                                "In round 2 no model of ours was scored on the test record before this freeze."),
    }
    if not cfg["identical_to_ct_phys8"]:
        cfg["gate"] = {"train_rmse_seed0": fin["dry_run"]["train_rmse"], "tol": 0.003, "source": "dry run in the selection kernel (full estimation record, seed 0, final hyper-parameters)"}
    else:
        cfg["final_run"] = "none: the selected configuration is numerically identical to ct_phys8 (see identity_check); TEST-LOG.md rows 15-24 stand and no test record is touched again"
    json.dump(cfg, open(a.out, "w"), indent=1)
    print("wrote", a.out, "| structure", fin["structure"], "| fit", fin["variant"], "| identical:", cfg["identical_to_ct_phys8"])


if __name__ == "__main__":
    main()
