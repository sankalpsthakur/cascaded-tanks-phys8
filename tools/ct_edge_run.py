"""Edge numbers for a frozen Cascaded Tanks physics model. TRAIN record only: this script never touches the test record.

  * refits the frozen configuration on the training record (same code and seed as ct/final.py; deterministic) to get the
    parameters and the training-record initial state
  * parameter count and bytes on disk as fp32 / fp16 / int8 (per-tensor symmetric) via ct.edge.size_report
  * training-record RMSE with the fp32 / fp16 / int8-dequantised parameter vector (JAX float64 simulator)
  * the edge-ready variant: the C99 step emitted by ct.edge.emit_c, compiled with gcc/cc/clang using -std=gnu99
    (ct.edge.build_and_bench hardcodes `clang -std=c99`, under which glibc hides clock_gettime), checked against the
    JAX simulation on the TRAIN input, timed in ns/step (ct.edge's own --bench loop, which includes one sin() per step,
    and a step-only harness with precomputed inputs)
  * pure-Python float step latency (ct.edge.python_step_latency)

usage: python tools/ct_edge_run.py results/FREEZE_ct_phys8.json --seed 0 --out edge_out
Latency numbers from a Kaggle kernel are Linux x86 numbers; the M4 number is measured separately from ct_step.c + bench_main.c.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import edge, phys  # noqa: E402
from ct import protocol as P  # noqa: E402
from ct.final import build_structure  # noqa: E402

HARNESS = r"""// step-only timing harness: precomputed inputs, no libm call inside the timed loop. Build: cc -O2 -std=gnu99 -o ct_bench bench_main.c -lm
#include "ct_step.c"
#include <stdlib.h>
int main(int argc, char **argv) {
  enum { N = 1024 };
  const int reps = argc > 1 ? atoi(argv[1]) : 20000;
  static double u[N];
  for (int t = 0; t < N; t++) u[t] = 3.0 + 2.0 * sin(0.013 * t);
  const ct_state s0 = { %(x1)r, %(x2)r, 3.0, 3.0 };
  double acc = 0.0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (int r = 0; r < reps; r++) {
    ct_state s = s0;
    for (int t = 0; t < N; t++) { ct_step(&s, u[t]); acc += s.x2; }
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)reps * N);
  printf("%%.3f ns_per_step (step only, checksum %%.6f)\n", ns, acc);
  return 0;
}
"""


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def find_cc():
    for c in ("gcc", "cc", "clang"):
        p = shutil.which(c)
        if p:
            return p
    return None


def cc_version(cc):
    try:
        return subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("freeze")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edge_out")
    a = ap.parse_args()
    cfg = json.load(open(a.freeze))
    struct = build_structure(cfg)
    fit = cfg["fit"]
    a.out = os.path.abspath(a.out)
    os.makedirs(a.out, exist_ok=True)

    u, y, Ts = P.load_train()
    uh = np.array([u[0], u[0]])
    fitter = phys.Fitter(struct)
    t0 = time.time()
    z, loss = fitter.fit(u, y, n_starts=fit["n_starts"], seed=a.seed, maxiter=fit["maxiter"])
    th, x0 = fitter.theta(z)
    out = dict(name=cfg["name"], seed=a.seed, fit_secs=time.time() - t0, train_rmse=float(np.sqrt(loss)), theta=th, x0_train=[float(v) for v in x0],
               host=dict(platform=platform.platform(), cpu=cpu_model(), cpu_count=os.cpu_count(), python=sys.version.split()[0]))

    # same fit as in the frozen final run? (deterministic fit; the stored theta is from results/ct_final_<name>.json if present)
    fin = os.path.join(ROOT, "results", f"ct_final_{cfg['name']}.json")
    if os.path.exists(fin):
        st = json.load(open(fin))["seeds"].get(str(a.seed))
        if st:
            out["theta_max_abs_diff_vs_final_run"] = float(max(abs(st["theta"][k] - th[k]) for k in struct.free))
            out["train_rmse_final_run"] = st["train_rmse"]

    # ---- sizes
    sr = edge.size_report(th, x0, struct.free, os.path.join(a.out, "params"))
    vec = sr["_vec"]
    out["n_params_free"] = sr["n_params_free"]
    out["n_params_incl_x0"] = sr["n_params_incl_x0"]
    out["bytes"] = {k: sr[k] for k in ("fp32_bytes", "fp16_bytes", "int8_bytes")}
    out["bytes_note"] = "np.save/np.savez containers (about 128 B of .npy header each); raw payload = 4/2/1 bytes per number (+4 B scale for int8)"
    out["raw_payload_bytes"] = dict(fp32=4 * int(vec.size), fp16=2 * int(vec.size), int8=int(vec.size) + 4)

    # ---- quantisation effect on the TRAIN record
    import jax.numpy as jnp

    def sim_with(th_d, x0v):
        yh, _ = fitter.sim({k: jnp.asarray(float(v)) for k, v in th_d.items()}, jnp.asarray(np.asarray(x0v, float)), jnp.asarray(u), jnp.asarray(uh))
        return np.asarray(yh)

    def variant(deq):
        thv = dict(th)
        for i, k in enumerate(struct.free):
            thv[k] = float(deq[i])
        return thv, np.asarray(deq[len(struct.free):], float)

    rm_all = lambda yh: float(np.sqrt(np.mean((yh - y) ** 2)))
    rm_50 = lambda yh: float(np.sqrt(np.mean((yh[P.N_INIT:] - y[P.N_INIT:]) ** 2)))
    y64 = sim_with(th, x0)
    q = {}
    for name, deq in (("fp64", None), ("fp32", sr["_deq32"]), ("fp16", sr["_deq16"]), ("int8_per_tensor", sr["_deq8"])):
        if deq is None:
            yh, rel = y64, 0.0
        else:
            thv, x0v = variant(deq)
            yh = sim_with(thv, x0v)
            rel = float(np.max(np.abs(deq - vec) / np.maximum(np.abs(vec), 1e-300)))
        q[name] = dict(train_rmse=rm_all(yh), train_rmse_from_sample50=rm_50(yh), delta_vs_fp64=rm_all(yh) - rm_all(y64),
                       max_abs_dev_of_trajectory_vs_fp64=float(np.max(np.abs(yh - y64))), max_relative_parameter_error=rel,
                       finite=bool(np.all(np.isfinite(yh))))
    out["quantisation_train_record"] = q

    # ---- C99 edge-ready variant
    src = edge.emit_c(th, x0, struct)
    cpath = os.path.join(a.out, "ct_step.c")
    open(cpath, "w").write(src)
    open(os.path.join(a.out, "bench_main.c"), "w").write(HARNESS % dict(x1=float(x0[0]), x2=float(x0[1])))
    out["c99"] = dict(bytes_source=os.path.getsize(cpath))
    cc = find_cc()
    if cc is None:
        out["c99"]["error"] = "no C compiler in this environment"
    else:
        out["c99"].update(compiler=cc_version(cc), flags="-O2 -std=gnu99")
        exe = os.path.join(a.out, "ct_main")
        subprocess.check_call([cc, "-O2", "-std=gnu99", "-DCT_MAIN", "-o", exe, cpath, "-lm"])
        # plain Python floats: with NumPy >= 2, repr(np.float64) is 'np.float64(..)', which the C reader cannot parse
        yc = edge.c_simulate(exe, [float(v) for v in x0], u, [float(v) for v in uh])
        out["c99"]["train_max_abs_diff_vs_jax"] = float(np.max(np.abs(yc - y64)))
        out["c99"]["train_rmse_c99"] = rm_all(yc)
        med, vals = edge.c_bench(exe, reps=3)
        out["c99"]["ns_per_step_bench_incl_sin"] = med
        out["c99"]["ns_per_step_bench_runs"] = vals
        hexe = os.path.join(a.out, "ct_bench")
        subprocess.check_call([cc, "-O2", "-std=gnu99", "-o", hexe, os.path.join(a.out, "bench_main.c"), "-lm"])
        runs = [float(subprocess.run([hexe], capture_output=True, text=True, check=True).stdout.split()[0]) for _ in range(5)]
        out["c99"]["ns_per_step_step_only"] = statistics.median(runs)
        out["c99"]["ns_per_step_step_only_runs"] = runs
        out["c99"]["one_step_covers"] = "one 4-s sample = 4 RK4 sub-steps (16 right-hand-side evaluations, 2 powers/sqrt each) + overflow clamp"
    out["python_float_ns_per_step"] = edge.python_step_latency(th, x0, struct, n=5000)
    out["reference_0221_entry"] = "see results/ct_reference_edge.json (>= 330 parameters, 222 us/step in torch eager on a loaded Mac, lower bound)"
    json.dump(out, open(os.path.join(ROOT, "results", f"ct_edge_{cfg['name']}.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("train_rmse", "n_params_incl_x0", "bytes", "raw_payload_bytes")}, indent=1))
    print(json.dumps(out["quantisation_train_record"], indent=1))
    print(json.dumps(out["c99"], indent=1))
    print("python float ns/step:", out["python_float_ns_per_step"])


if __name__ == "__main__":
    main()
