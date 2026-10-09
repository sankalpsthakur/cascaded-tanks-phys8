"""Edge numbers for the round-2 frozen Cascaded Tanks model. TRAIN record only: this script never touches the test record.

  * parameters of one seed from results/ct_final_<name>.json (the fit of ct2/final2.py; no refit), parameter count and bytes as fp32 / fp16 / int8 (per-tensor symmetric)
  * training-record RMSE with the fp32 / fp16 / int8-dequantised parameter vector (JAX float64 simulator of ct2/phys2.py)
  * C99 edge-ready step and timing, only when the structure is expressible by ct/edge.py's emitter (no h0, tp, to, w3); otherwise reported as not applicable

usage: python tools/ct2_edge_run.py results/FREEZE_ct_round2.json --seed 0 --out edge_out
"""
from __future__ import annotations

import argparse, json, os, shutil, statistics, subprocess, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from ct import edge, phys  # noqa: E402  (round-1 modules, unchanged)
from ct import protocol as P  # noqa: E402
from ct2 import phys2  # noqa: E402
from ct2.final2 import fitter_from  # noqa: E402
import ct_edge_run as R1  # noqa: E402  (harness text, compiler helpers of the round-1 edge script)

EMITTER_OK = {"k1", "r2", "k3", "k4", "beta", "a", "b4", "cA", "qov", "c4", "cA1", "d1", "d2", "p1", "p3", "c2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freeze")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edge_out")
    ap.add_argument("--results", default="", help="results json of ct2/final2.py (default results/ct_final_<name>.json)")
    ap.add_argument("--dest", default="", help="output json (default results/ct_edge_<name>.json)")
    a = ap.parse_args()
    cfg = json.load(open(a.freeze))
    name = cfg["name"]
    res = json.load(open(a.results or os.path.join(ROOT, "results", f"ct_final_{name}.json")))
    z = np.array(res["seeds"][str(a.seed)]["z"])
    f = fitter_from(cfg)
    free = list(f.s.free)
    th, x0 = f.theta(z)
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)
    u, y, Ts = P.load_train()
    uh = np.array([u[0], u[0]])
    out = dict(name=name, seed=a.seed, structure=cfg["structure"]["id"], free=free, theta={k: th[k] for k in free}, x0_train=[float(v) for v in x0],
               train_rmse_in_results=res["seeds"][str(a.seed)]["train_rmse"], host=dict(platform=sys.platform, cpu=R1.cpu_model(), cpu_count=os.cpu_count()))

    sr = edge.size_report(th, x0, free, os.path.join(a.out, "params"))
    vec = sr["_vec"]
    out["n_params_free"], out["n_numbers_incl_x0"] = sr["n_params_free"], sr["n_params_incl_x0"]
    out["bytes_npy_containers"] = {k: sr[k] for k in ("fp32_bytes", "fp16_bytes", "int8_bytes")}
    out["raw_payload_bytes"] = dict(fp32=4 * int(vec.size), fp16=2 * int(vec.size), int8=int(vec.size) + 4)

    import jax.numpy as jnp

    def sim_with(deq):
        thv = {k: jnp.asarray(float(v)) for k, v in th.items()}
        for i, k in enumerate(free):
            thv[k] = jnp.asarray(float(deq[i]))
        x0v = jnp.asarray(np.asarray(deq[len(free):], float))
        return np.asarray(f.sim(thv, x0v, jnp.asarray(u), jnp.asarray(uh))[0])

    rm = lambda yh: float(np.sqrt(np.mean((yh - y) ** 2)))
    y64 = sim_with(vec)
    q = {}
    for nm, deq in (("fp64", vec), ("fp32", sr["_deq32"]), ("fp16", sr["_deq16"]), ("int8_per_tensor", sr["_deq8"])):
        yh = y64 if nm == "fp64" else sim_with(deq)
        q[nm] = dict(train_rmse=rm(yh), delta_vs_fp64=rm(yh) - rm(y64), max_abs_dev_of_trajectory_vs_fp64=float(np.max(np.abs(yh - y64))),
                     max_relative_parameter_error=float(np.max(np.abs(deq - vec) / np.maximum(np.abs(vec), 1e-300))), finite=bool(np.all(np.isfinite(yh))))
    out["quantisation_train_record"] = q

    if not set(free) <= EMITTER_OK:
        out["c99"] = dict(applicable=False, reason="ct/edge.py's C99 emitter does not cover " + ", ".join(sorted(set(free) - EMITTER_OK)))
    else:
        struct1 = phys.Structure(free=tuple(free), tie_r2=False, clamp_x2=True, nsub=4)
        src = edge.emit_c(th, x0, struct1)
        cpath = os.path.join(a.out, "ct_step.c")
        open(cpath, "w").write(src)
        open(os.path.join(a.out, "bench_main.c"), "w").write(R1.HARNESS % dict(x1=float(x0[0]), x2=float(x0[1])))
        out["c99"] = dict(applicable=True, bytes_source=os.path.getsize(cpath))
        cc = R1.find_cc()
        if cc is None:
            out["c99"]["error"] = "no C compiler"
        else:
            out["c99"].update(compiler=R1.cc_version(cc), flags="-O2 -std=gnu99")
            exe = os.path.join(a.out, "ct_main")
            subprocess.check_call([cc, "-O2", "-std=gnu99", "-DCT_MAIN", "-o", exe, cpath, "-lm"])
            yc = edge.c_simulate(exe, [float(v) for v in x0], u, [float(v) for v in uh])
            out["c99"]["train_max_abs_diff_vs_jax"] = float(np.max(np.abs(yc - y64)))
            out["c99"]["train_rmse_c99"] = rm(yc)
            hexe = os.path.join(a.out, "ct_bench")
            subprocess.check_call([cc, "-O2", "-std=gnu99", "-o", hexe, os.path.join(a.out, "bench_main.c"), "-lm"])
            runs = [float(subprocess.run([hexe], capture_output=True, text=True, check=True).stdout.split()[0]) for _ in range(5)]
            out["c99"]["ns_per_step_step_only"] = statistics.median(runs)
            out["c99"]["ns_per_step_step_only_runs"] = runs
        out["python_float_ns_per_step"] = edge.python_step_latency(th, x0, struct1, n=5000)
    json.dump(out, open(a.dest or os.path.join(ROOT, "results", f"ct_edge_{name}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("structure", "n_params_free", "n_numbers_incl_x0", "raw_payload_bytes")}, indent=1))
    print(json.dumps(out["quantisation_train_record"], indent=1))
    print(json.dumps(out["c99"], indent=1))


if __name__ == "__main__":
    main()
