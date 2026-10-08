# Cascaded Tanks free-run simulation with a 10-number grey-box model (`ct_phys8`)

Technical note. Self-evaluated on 2026-10-07; not peer-reviewed. Author: Sankalp Thakur.

**Summary.** The frozen model scores a mean test RMSE of 0.2719 V over ten optimiser seeds on the official split (n_init = 50, free-run simulation); the seeds all reach the same optimum (train RMSE 0.17848, test RMSE 0.2717-0.2720). The 95 % moving-block-bootstrap interval of the RMSE is 0.216-0.339 V.
It does not beat the 0.221 V entry. It is within the seed spread of that entry's own code (0.2163 at the notebook default seed 42, mean 0.2759 over seeds 0-9, maximum 0.3465).

## 1. Protocol

- Data: `nonlinear_benchmarks` 1.0.1 loader, official split (estimation record 1024 samples, test record 1024 samples, Ts = 4 s, state-initialisation window of 50 samples). Metric: `nonlinear_benchmarks.error_metrics.RMSE(y_test, y_model, n_init=50)`, volts. Data license CC BY-SA 4.0; the data is not redistributed here.
- The loader's default URL (4TU.ResearchData) returns an HTML page and the loader caches that page, which masks its own Google-Drive fallback; `ct/protocol.py` passes the Drive URL explicitly. The final run downloaded the data from that URL inside the kernel and checked the sha256 of the zip (`eb0fa058...51b`) and of `dataBenchmark.mat` (`cb2f88d4...eab`); both matched.
- Selection used the estimation record only (section 3). The configuration was frozen and committed (`results/FREEZE_ct_phys8.json`; the claim rule was appended before any score of ours existed) before the test record was scored with our model. Every touch of the test record is in `TEST-LOG.md`:
  rows 1-14 before the freeze (reproduction of third-party code, crashed attempts, one disclosure), rows 15-24 the ten frozen runs, each seed scored exactly once.
- Disclosure: before the freeze, summary statistics of the test record (min/max of y, first samples, number of saturated samples) were printed once at load time and a third-party model (the 0.221 entry's code) was run on it. The test minimum (2.16 V) lies below the estimation minimum (2.91 V);
  this motivated an additional validation on the estimation record that masks its lowest 12 % of outputs. No model of ours was scored on the test record before the freeze.

## 2. Model (`ct/phys.py`)

Two cascaded Torricelli tanks, time unit = one sample, RK4 with 4 sub-steps, zero-order-hold input, levels in sensor volts:

```
q1  = k1 sqrt(x1)                          upper-tank outflow
dx1 = k4 (u - b4) - q1                     pump with a signed offset b4
dx2 = (r2 q1 - k3 sqrt(x2)) / max(1 + cA (x2 - 5), 0.2)     bottle-shaped lower tank (capacity slope cA)
overflow: s = max(x1 - 10, 0); x1 <- x1 - s; x2 <- x2 + beta s; x2 <- min(x2, 10 - a)
y   = a + x2   (clipped at 10 V)
```

Free parameters (8): k1, r2, k3, k4, beta, a, b4, cA. Fitted values (seed 0): k1 0.15919, r2 0.72476, k3 0.12574, k4 0.25084, beta 0.64028, a 1.80947, b4 1.49031, cA -0.17332. The two initial states are fitted per record from the first 50 samples only, with the eight parameters frozen.
Training: simulation-error MSE over the whole estimation record, L-BFGS-B on JAX float64 gradients, 8 random starts, 600 iterations, best start by loss. No neural component, no data-driven residual.

## 3. Selection on the estimation record only

Four contiguous blocks of 256 samples; a block is masked, the model is fitted on the rest and the block is scored like the official test (window initialisation, free run, RMSE over samples 50..255). Pooled CV RMSE:

| structure | pooled CV RMSE [V] |
|---|---|
| A base (k1, r2, k3, k4, beta, a) | 0.307 |
| B + pump offset b4 | 0.262 |
| **C + capacity slope cA (= `ct_phys8`)** | **0.209** |
| F1 + input FIR | 0.208 |
| R1 + residual MLP(4) | 0.226 |
| H1 + upper outlet height | 0.234 |
| Z1 dead-zone pump | worse than C |
| E free exponents | overfits (fold 0: 0.215 vs 0.139) |
| D sensor curvature | unstable window fit |

Rule: the smallest structure within one standard error (about 0.025) of the best pooled CV RMSE. Level-extrapolation check (lowest 12 % of the outputs masked and scored there): A 0.663, B 0.613, C 0.396.

## 4. Results (kernel `sankalpsthakur/bench-control-ct-final` v1, 2026-10-07 10:12-11:08 UTC, private)

| seed | test RMSE [V] | train RMSE [V] | 95 % block-bootstrap interval of this seed's RMSE |
|---|---|---|---|
| 0 | 0.2720 | 0.1785 | 0.2165 .. 0.3395 |
| 1 | 0.2719 | 0.1785 | 0.2164 .. 0.3393 |
| 2 | 0.2720 | 0.1785 | 0.2164 .. 0.3395 |
| 3 | 0.2719 | 0.1785 | 0.2164 .. 0.3394 |
| 4 | 0.2717 | 0.1785 | 0.2165 .. 0.3389 |
| 5 | 0.2718 | 0.1785 | 0.2165 .. 0.3391 |
| 6 | 0.2718 | 0.1785 | 0.2164 .. 0.3391 |
| 7 | 0.2719 | 0.1785 | 0.2164 .. 0.3393 |
| 8 | 0.2718 | 0.1785 | 0.2165 .. 0.3391 |
| 9 | 0.2720 | 0.1785 | 0.2165 .. 0.3395 |
| **mean** | **0.27189** (sd over seeds 0.0001) | 0.17848 | **0.2164 .. 0.3393** (interval of the mean statistic, same resampled blocks for every seed) |

The intervals are moving-block bootstraps (block 25, 4000 draws) of the test residuals; they are wide because the scored record has 974 strongly autocorrelated samples. The seeds differ only in the random starts of the optimiser, so they are not independent replications of the data.
The largest error (-1.00 V at sample 386: 7.39 V predicted against 8.39 V measured) is on the fast rise to the second large peak, which the model also under-shoots (9.11 V against 9.70 V at sample 408); 86 % of the squared error comes from the 352 of 974 scored samples with an absolute error above 0.25 V (seed 0; `results/figs/ct_ct_phys8_test.png`).

Paired comparison with the third-party 0.221 entry's code (11 seeds, rerun by us, `ct/compare_ref.py`, block 25, same resampled indices for both models; our seed 0 shown, the other nine seeds differ by less than 0.0005 V):

| reference seed | its RMSE | ours minus reference | 95 % paired interval |
|---|---|---|---|
| 42 (notebook default) | 0.2163 | +0.0557 | -0.0025 .. +0.1232 |
| 3 | 0.2383 | +0.0337 | -0.0188 .. +0.0982 |
| 8 | 0.2400 | +0.0320 | -0.0374 .. +0.1115 |
| 9 | 0.2469 | +0.0251 | -0.0394 .. +0.1011 |
| 0 | 0.2516 | +0.0204 | -0.0531 .. +0.1041 |
| 7 | 0.2561 | +0.0159 | -0.0417 .. +0.0821 |
| 1 | 0.2658 | +0.0062 | -0.0490 .. +0.0700 |
| 5 | 0.2795 | -0.0075 | -0.0859 .. +0.0787 |
| 2 | 0.3093 | -0.0373 | -0.1165 .. +0.0511 |
| 4 | 0.3245 | -0.0525 | -0.1260 .. +0.0272 |
| 6 | 0.3465 | -0.0745 | -0.1530 .. +0.0110 |

No interval excludes zero: the comparison is inconclusive in both directions (fraction of the other reference seeds beaten under the pre-registered rule: 0 of 10).

## 5. Edge numbers

Kernel `sankalpsthakur/bench-control-ct-edge` v1, 2026-10-07, train record only; the kernel's refit reproduced the final run's parameters exactly (max difference 0.0, train RMSE 0.178476).

- Model size: 10 numbers (the 8 free parameters and the two initial states of a record; at deployment the initial state is re-fitted from the first 50 samples of the record, which is not part of the step timings below). Parameter payload 40 B in fp32, 20 B in fp16, 14 B in int8 (10 values + a 4-byte scale); as `.npy`/`.npz` containers 168 / 148 / 512 B.
- Edge-ready variant: `ct_step.c` (3223 B of C99 source, double precision, no allocation, 4 RK4 sub-steps plus the overflow clamp, emitted by `ct/edge.py`). On the train input its output differs from the JAX simulation by at most 5.0e-13 V. Speed on one core of an Intel Xeon @ 2.20 GHz (Kaggle container, gcc 13.3 -O2): **289 ns per 4-s sample** (median of 5 step-only runs; 296 ns with the input generator of `ct/edge.py`); the same step in pure Python floats takes 28.7 us.
- Quantisation, effect on the train-record RMSE (float64: 0.17848 V): fp32 +1e-10 V; fp16 0.17850 V (+2.6e-5 V, largest trajectory deviation 0.0086 V); **int8 with one shared scale fails**: 0.742 V (+0.564 V; parameter errors up to 7.5 %, the small parameters suffer most). fp16 is safe, int8 per tensor is not.
- Apple M4 latency: deferred (the machine was busy during the measurement window); `tools/m4_latency.sh ct results/ct_edge_partial` is ready and runs only below a 1-minute load average of 4.
- Comparison model (the 0.221 entry's code): at least 330 parameters (at least 1320 B in fp32) and 222 us per step for its network components in torch eager on a loaded Mac (a lower bound). It was not measured on the same machine or runtime as the C99 step, so no speed ratio is claimed.

## 6. Reference-code observations

The code of the entry that sets the 0.221 V mark (github.com/MiladBnit/GreyBox-Neural-State-Space-Normalization @ 5f975bad) was run unchanged with the notebook defaults, 11 seeds, scored with the official metric (`TEST-LOG.md` rows 1-13):
seed 42 gives 0.2163 and the other ten seeds 0.2383-0.3465 (mean 0.2759); only seed 42 is below 0.221. Its pipeline uses the test record as its validation set and starts the test simulation from the training initial state (the first 6 test samples are dropped):
at commit 5f975bad, `user_config.py` lines 75-78 build `u_val`, `y_val` from the `u_test`, `y_test` columns of `dataBenchmark.csv` (all with `[6:]`), and `Core.py` lines 685-703 run the physical simulator on the training, validation and test inputs from the same fitted initial state `x0_phys`.
Size of that model: at least 330 parameters (321 neural), 222 us per step for its components in torch eager on a loaded Mac (a lower bound for the full simulator; `results/ct_reference_edge.json`).

## 7. Limitations

One physical structure, one dataset. The interval is a block bootstrap of one test record, not an estimate for other records. The seed spread is not a data-uncertainty estimate. Kaggle latency numbers are Linux x86, not the M4.
The model is a close relative of the 2017 grey-box overflow model on the sheet's legacy rows (Rogers, Holmes, Cross, Worden 2017, 0.18 V on the pre-official split, listed as legacy), so novelty is limited to the capacity-slope term, the pump offset and the protocol (frozen configuration, test log, paired comparison).

## 8. Reproduction

`kernels/ct_final/make_kernel.py` builds the kernel script from the committed sources (a sha256 manifest of every payload file is inside); the kernel recomputes the git tree hashes of `ct/` and of the freeze file and aborts on a mismatch, checks the data checksums, reproduces the local train-only gate (0.1785 +- 0.003) and only then scores the test record.
The outputs are in `logs/kaggle-output/ct-final/` (log, per-stage logs, `manifest_verified.json`, `versions.json`, `venv_freeze.txt`, predictions, results) and `results/ct_*_ct_phys8.json`.
