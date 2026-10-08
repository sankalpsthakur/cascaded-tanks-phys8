# Cascaded Tanks: a 10-number grey-box model (`ct_phys8`)

An entry for the [Cascaded Tanks](https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks)
system-identification benchmark: two cascaded Torricelli tanks with overflow routing, a
bottle-shaped lower tank, a signed pump offset and a sensor offset. Eight free physical
parameters, plus two initial states fitted per record from the first 50 samples. No neural or
data-driven component. The technical note is [REPORT.md](REPORT.md).

## Result (official split, free-run simulation, n_init = 50; self-evaluated 2026-10-07)

- Mean test RMSE over ten optimiser seeds: **0.2719 V** (seeds 0.2717-0.2720; they converge to the same optimum).
- 95 % moving-block-bootstrap interval of the RMSE: **0.216-0.339 V**.
- It does **not** beat the 0.221 V entry on the results sheet. A paired block bootstrap against that entry's own code is inconclusive in both directions (`results/ct_compare_ref_ct_phys8.json`).
- Prior art: Rogers, Holmes, Cross and Worden (2017), a grey-box model with a physical overflow term, 0.18 V on the pre-official split (listed as legacy on the sheet).

What the reference code shows: the 0.221 V entry's code, run unchanged with the notebook defaults, scores 0.2163 at its default seed 42 and 0.2383-0.3465 over seeds 0-9 (mean 0.2759). Its pipeline uses the test record as its validation set. Details in REPORT.md section 6 and `TEST-LOG.md` rows 1-13.

## Edge numbers

10 numbers: 40 B in fp32, 20 B in fp16, 14 B in int8. fp16 changes the train RMSE by 2.6e-5 V; int8 with one shared scale fails (+0.56 V). The C99 step (`results/ct_edge_partial/ct_step.c`, emitted by `ct/edge.py`) takes 289 ns per 4-s sample on one Intel Xeon core at 2.2 GHz and matches the JAX simulation within 5e-13 V. Apple M4 latency: not measured yet (`tools/m4_latency.sh`).

## Integrity

- Structure selected by blocked cross-validation on the estimation record only (REPORT.md section 3).
- Configuration frozen and committed (`results/FREEZE_ct_phys8.json`, with a claim rule) before the test record was scored with this model.
- Every touch of the test record is a row in `TEST-LOG.md`: 11 reference-code seeds, two crashed attempts, one disclosure, then the ten frozen runs (each seed scored once).
- The frozen `ct/` directory in this repository hashes to git tree `120f81f08ff5460b0a8d79758ebc28364ba9eca7` and the freeze file to blob `55e4323c1d28d833c90aa1f7d97da32c08fb5413`; the final kernel recomputes both and aborts on a mismatch.

## Reproduce

Data: the `nonlinear_benchmarks` 1.0.1 loader's default 4TU URL returns an HTML page; `ct/protocol.py` passes the loader's Google-Drive fallback URL. The data (CC BY-SA 4.0) is not redistributed here; the final run checked its sha256.

On Kaggle (what produced the numbers): `kernels/ct_final/make_kernel.py` builds the kernel from the committed sources with a sha256 manifest; push it as a private kernel. Its outputs are in `logs/kaggle-output/ct-final/`.

Locally (about 2 minutes per seed, one thread), in an environment with JAX (float64), SciPy and `nonlinear_benchmarks`:

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
python ct/final.py results/FREEZE_ct_phys8.json --seeds 0 1 2 3 4 5 6 7 8 9
python ct/compare_ref.py ct_phys8
python ct/plots.py ct_phys8
```

`ct/repro_reference.py --seed 42` reruns the 0.221 V entry's code; `ct/cv2.py C0` reproduces the selection evidence.

## Licence

Apache-2.0 (see `LICENSE`). Sankalp Thakur, October 2026. Self-evaluated; not peer-reviewed.
