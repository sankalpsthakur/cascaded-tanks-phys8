"""Edge numbers for a frozen physics-structured Cascaded Tanks model.

* parameter count, size on disk (fp32 / fp16 / int8 per-tensor) of the parameter file
* C99 single-step implementation (RK4 sub-steps + overflow clamp), compiled with clang -O2, ns per sample at 1 thread
* pure-Python (float) step latency, and the JAX reference for verification
* numerical agreement C99 vs JAX on a TRAIN-record input (the test record is not used here)
"""
from __future__ import annotations

import json, os, subprocess, sys, tempfile, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ct import phys  # noqa: E402


def _fmt(v):
    return repr(float(v))


def emit_c(th: dict, x0, struct: phys.Structure, nsamples=1024):
    assert struct.res_hidden == 0, "C emitter covers the pure-physics structures"
    use_fir = "d1" in struct.free or "d2" in struct.free
    e = np.exp([0.0, th["d1"], th["d2"]]) if use_fir else np.array([1.0, 0.0, 0.0])
    w = e / e.sum()

    def powc(x, p):
        return f"sqrt({x})" if abs(p - 0.5) < 1e-15 else f"pow({x}, {_fmt(p)})"

    dz = "uu = 0.08 * log1p(exp(uu / 0.08));" if struct.deadzone else ""
    src = f"""// Auto-generated Cascaded Tanks step (frozen parameters). C99, double precision.
#include <math.h>
#include <stdio.h>
#include <time.h>
#define EPS {_fmt(phys.EPS)}
#define XM1 {_fmt(phys.XM1)}
#define YCLIP {_fmt(phys.Y_CLIP)}
#define NSUB {struct.nsub}
static const double K1={_fmt(th['k1'])}, R2={_fmt(th['r2'])}, K3={_fmt(th['k3'])}, K4={_fmt(th['k4'])}, BETA={_fmt(th['beta'])};
static const double A={_fmt(th['a'])}, B4={_fmt(th['b4'])}, P1={_fmt(th['p1'])}, P3={_fmt(th['p3'])}, C2={_fmt(th['c2'])}, CA={_fmt(th['cA'])}, QOV={_fmt(th['qov'])}, C4={_fmt(th['c4'])}, CA1={_fmt(th['cA1'])};
static const double W0={_fmt(w[0])}, W1={_fmt(w[1])}, W2={_fmt(w[2])};
typedef struct {{ double x1, x2, um1, um2; }} ct_state;   // um1,um2: previous inputs (input FIR)
static inline void rhs(double x1, double x2, double u, double *d1, double *d2) {{
  x1 = x1 > 0 ? x1 : 0; x2 = x2 > 0 ? x2 : 0;
  double q1 = K1 * {powc('(x1 + EPS)', th['p1'])};
  double cap = 1.0 + CA * (x2 - 5.0); cap = cap > 0.2 ? cap : 0.2;
  double uu = u - B4; {dz}
  double cap1 = 1.0 + CA1 * (x1 - 5.0); cap1 = cap1 > 0.2 ? cap1 : 0.2;
  *d1 = (K4 * (uu + C4 * uu * uu) - q1) / cap1;
  *d2 = (R2 * q1 - K3 * {powc('(x2 + EPS)', th['p3'])}) / cap;
}}
static inline void clampx(double *x1, double *x2) {{
  double sur = *x1 - XM1; sur = sur > 0 ? sur : 0;
  const double hh = 1.0 / NSUB;
  *x1 -= sur; *x2 += {"BETA * sur" if abs(th['qov']-1.0) < 1e-15 else "BETA * hh * (pow(sur / hh + 1e-6, QOV) - pow(1e-6, QOV))"};
  {"double mx = YCLIP - A; if (*x2 > mx) *x2 = mx;" if struct.clamp_x2 else ""}
  if (*x1 < 0) *x1 = 0; if (*x2 < 0) *x2 = 0;
}}
// advance one sample with zero-order-hold input u; returns nothing, updates state
static inline void ct_step(ct_state *s, double u) {{
  double ue = W0 * u + W1 * s->um1 + W2 * s->um2;
  const double h = 1.0 / NSUB;
  double x1 = s->x1, x2 = s->x2, a1, a2, b1, b2, c1, c2_, d1, d2;
  for (int i = 0; i < NSUB; i++) {{
    rhs(x1, x2, ue, &a1, &a2);
    rhs(x1 + 0.5*h*a1, x2 + 0.5*h*a2, ue, &b1, &b2);
    rhs(x1 + 0.5*h*b1, x2 + 0.5*h*b2, ue, &c1, &c2_);
    rhs(x1 + h*c1, x2 + h*c2_, ue, &d1, &d2);
    x1 += (h/6.0) * (a1 + 2*b1 + 2*c1 + d1);
    x2 += (h/6.0) * (a2 + 2*b2 + 2*c2_ + d2);
    clampx(&x1, &x2);
  }}
  s->x1 = x1; s->x2 = x2; s->um2 = s->um1; s->um1 = u;
}}
static inline double ct_out(const ct_state *s) {{
  double y = A + s->x2 + C2 * s->x2 * s->x2; return y < YCLIP ? y : YCLIP;
}}
#ifdef CT_MAIN
int main(int argc, char **argv) {{
  // usage: ct_main  -> reads N, x1, x2, u[-2], u[-1] then N inputs from stdin, writes N outputs; or --bench
  int n; double x1, x2, um2, um1;
  if (argc > 1 && argv[1][0]=='-') {{
    int reps = 20000; n = {nsamples}; double acc = 0;
    ct_state s = {{ {_fmt(x0[0])}, {_fmt(x0[1])}, 3.0, 3.0 }};
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < reps; r++) {{
      s.x1 = {_fmt(x0[0])}; s.x2 = {_fmt(x0[1])};
      for (int t = 0; t < n; t++) {{ double u = 3.0 + 2.0 * sin(0.013 * t + r); ct_step(&s, u); acc += ct_out(&s); }}
    }}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ((double)reps * n);
    printf("%.3f ns_per_step (checksum %.6f)\\n", ns, acc);
    return 0;
  }}
  if (scanf("%d %lf %lf %lf %lf", &n, &x1, &x2, &um2, &um1) != 5) return 1;
  ct_state s = {{ x1, x2, um1, um2 }};
  for (int t = 0; t < n; t++) {{ double u; if (scanf("%lf", &u) != 1) return 2; printf("%.12f\\n", ct_out(&s)); ct_step(&s, u); }}
  return 0;
}}
#endif
"""
    return src


def build_and_bench(src: str, workdir: str):
    os.makedirs(workdir, exist_ok=True)
    cpath = os.path.join(workdir, "ct_step.c")
    open(cpath, "w").write(src)
    exe = os.path.join(workdir, "ct_main")
    subprocess.check_call(["clang", "-O2", "-std=c99", "-DCT_MAIN", "-o", exe, cpath, "-lm"])
    return exe, cpath


def c_simulate(exe, x0, u, u_hist):
    inp = f"{len(u)} {x0[0]!r} {x0[1]!r} {u_hist[0]!r} {u_hist[1]!r}\n" + "\n".join(repr(float(v)) for v in u) + "\n"
    out = subprocess.run([exe], input=inp, capture_output=True, text=True, check=True).stdout.split()
    return np.array([float(v) for v in out])


def c_bench(exe, reps=3):
    vals = []
    for _ in range(reps):
        o = subprocess.run([exe, "--bench"], capture_output=True, text=True, check=True).stdout
        vals.append(float(o.split()[0]))
    return float(np.median(vals)), vals


def python_step_latency(th, x0, struct, n=20000):
    """Pure-Python scalar RK4 step (what a PLC-adjacent Python runtime would do)."""
    import math
    k1, r2, k3, k4, beta, a, b4, p1, p3, cA, qov, c4, cA1 = (th[k] for k in ("k1", "r2", "k3", "k4", "beta", "a", "b4", "p1", "p3", "cA", "qov", "c4", "cA1"))
    nsub = struct.nsub; h = 1.0 / nsub

    def rhs(x1, x2, u):
        x1 = max(x1, 0.0); x2 = max(x2, 0.0)
        q1 = k1 * (x1 + phys.EPS) ** p1
        cap = max(1.0 + cA * (x2 - 5.0), 0.2)
        uu = u - b4
        if struct.deadzone: uu = 0.08 * math.log1p(math.exp(uu / 0.08))
        cap1 = max(1.0 + cA1 * (x1 - 5.0), 0.2)
        return (k4 * (uu + c4 * uu * uu) - q1) / cap1, (r2 * q1 - k3 * (x2 + phys.EPS) ** p3) / cap

    def step(x1, x2, u):
        for _ in range(nsub):
            a1, a2 = rhs(x1, x2, u); b1, b2 = rhs(x1 + .5 * h * a1, x2 + .5 * h * a2, u)
            c1, c2 = rhs(x1 + .5 * h * b1, x2 + .5 * h * b2, u); d1, d2 = rhs(x1 + h * c1, x2 + h * c2, u)
            x1 += h / 6 * (a1 + 2 * b1 + 2 * c1 + d1); x2 += h / 6 * (a2 + 2 * b2 + 2 * c2 + d2)
            sur = max(x1 - phys.XM1, 0.0); x1 -= sur; x2 += beta * sur if qov == 1.0 else beta * h * ((sur / h + 1e-6) ** qov - 1e-6 ** qov)
            if struct.clamp_x2: x2 = min(x2, phys.Y_CLIP - a)
            x1 = max(x1, 0.0); x2 = max(x2, 0.0)
        return x1, x2

    x1, x2 = float(x0[0]), float(x0[1])
    ts = []
    for rep in range(5):
        t0 = time.perf_counter()
        for t in range(n):
            x1, x2 = step(x1, x2, 3.0 + 2.0 * math.sin(0.013 * t))
        ts.append((time.perf_counter() - t0) / n)
    return float(np.median(ts)) * 1e9


def size_report(th, x0, free, outdir):
    """Bytes on disk of the parameter vector as fp32 / fp16 / int8(per-tensor symmetric)."""
    vec = np.array([th[k] for k in free] + list(x0), float)
    os.makedirs(outdir, exist_ok=True)
    res = {"n_params_free": len(free), "n_params_incl_x0": int(vec.size)}
    p32 = os.path.join(outdir, "params_fp32.npy"); np.save(p32, vec.astype(np.float32)); res["fp32_bytes"] = os.path.getsize(p32)
    p16 = os.path.join(outdir, "params_fp16.npy"); np.save(p16, vec.astype(np.float16)); res["fp16_bytes"] = os.path.getsize(p16)
    sc = np.abs(vec).max() / 127.0
    q = np.round(vec / sc).astype(np.int8)
    p8 = os.path.join(outdir, "params_int8.npz"); np.savez(p8, q=q, scale=np.float32(sc)); res["int8_bytes"] = os.path.getsize(p8)
    res["_vec"] = vec; res["_deq32"] = vec.astype(np.float32).astype(float)
    res["_deq16"] = vec.astype(np.float16).astype(float); res["_deq8"] = q.astype(float) * sc
    return res
