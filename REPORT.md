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

---

## 9. Round 2 (2026-10-09): one more pre-registered shot

**Summary.** A second, pre-registered selection (14 candidate structures, estimation record only) chose `ct_phys8` plus a level-dependent lower-outlet coefficient (11 numbers); frozen and scored once per seed (10 seeds): mean test RMSE 0.2362 V, 95 % block-bootstrap interval 0.200-0.274 V (round 1: 0.2719 V). It does not beat the 0.221 V entry (paired difference to the entry's code at its default seed +0.0199 V, interval -0.0184 .. +0.0584; the pre-registered rule is not met). The candidate families were chosen after the round-1 test-residual analysis, so this is not an independent test of the record.

Written after the round-2 runs. The registration is `PREREGISTRATION-CT-ROUND2.md` (commit 0509f85, written before any round-2 fitting run; amendments and the record of what happened are in its section 12). Round-2 code is in `ct2/`; `ct/` is unchanged (git tree hash 120f81f0 in both kernels).

### 9.1 What was registered, and what the test record had already told us

- Candidates: 14 structures of at most 13 numbers, all `ct_phys8` plus one or more terms: first-order pump/inflow lag (`tp`), level-dependent lower-outlet coefficient (`w3`: q3 = k3 exp(w3 (x2 - 5)/5) sqrt(x2)), delayed return of the upper-tank overflow (`to`), flow-dependent overflow routing exponent (`qov`), pump curvature (`c4`), upper outlet height (`h0`), input FIR taps, upper capacity slope (`cA1`), three combinations, and a swap of the capacity slope for `w3`. Three fit variants (more random starts, pseudo-Huber loss, multiple shooting) were to be tried on the structure chosen first.
- Rule: the same four masked blocks of 256 as in round 1; pooled CV RMSE; the standard error of a candidate is `sd(block MSEs) / 2 / (2 x pooled RMSE)` (delta method; 0.026 for `ct_phys8`, which is round 1's "about 0.025"); the smallest structure within one standard error of the best pooled CV; ties by the level-extrapolation check (lowest 12 % of the outputs masked, scored there). Every candidate nested in `ct_phys8` got one extra start at `ct_phys8`'s optimum of the same job, so a richer model's training loss cannot end above `ct_phys8`'s because of an unlucky start. Claim rule: copied unchanged from round 1.
- Disclosure. Round 2 is not an independent test: the candidate families were chosen after the round-1 error analysis of the test residuals (section 4: 86 % of the squared error on the fast rise to the second large peak and its under-shoot; test minimum 2.16 V below the estimation minimum 2.91 V). Selection used the estimation record only, and the selection code cannot read the test record (loader patched to return `(train, None)`, `test_eval` replaced by a function that raises). No round-2 code read test values before the freeze, and the saved round-1 test predictions were not opened. nonlinearbenchmark.org's reporting rules forbid using test data to tune structure; a round-2 number is a second look at the same record with a test-informed candidate set and, whatever it is, must be read that way. Whether it may be sent anywhere is the lead's decision.

### 9.2 Selection on the estimation record (kernel `sankalpsthakur/bench-control-ct-round2-select` v1, 15:22-17:16 UTC, private)

Reproduction gate: `ct_phys8` (S00) re-run in the new code on Kaggle gave pooled CV 0.2091 (round 1: 0.2089; blocks 0.139 / 0.252 / 0.177 / 0.247) and level RMSE 0.396 (round 1: 0.396); the FIR and h0 candidates give 0.2082 and 0.2388 against round 1's F1 (0.2084) and H1 (0.2336).

| id | structure (added to ct_phys8) | numbers | block RMSE [V] (4 blocks) | pooled CV [V] | SE | level RMSE [V] | within best + 1 SE |
|---|---|---|---|---|---|---|---|
| S00 | C = ct_phys8 (reference) | 10 | 0.139 / 0.252 / 0.177 / 0.247 | 0.2091 | 0.0261 | 0.396 | no |
| S01 | +tp pump/inflow lag | 11 | 0.148 / 0.252 / 0.165 / 0.229 | 0.2030 | 0.0245 | 0.351 | yes |
| S02 | +w3 level-dependent lower outlet | 11 | 0.156 / 0.225 / 0.201 / 0.185 | 0.1934 | 0.0144 | 0.338 | **chosen** |
| S03 | +to delayed overflow return | 11 | 0.139 / 0.252 / 0.177 / 0.250 | 0.2100 | 0.0266 | 0.396 | no |
| S04 | +qov flow-dependent overflow routing | 11 | 0.140 / 0.252 / 0.178 / 0.234 | 0.2060 | 0.0246 | 0.396 | yes |
| S05 | +c4 pump curvature | 11 | 0.146 / 0.248 / 0.169 / 0.342 | 0.2389 | 0.0456 | 0.398 | no |
| S06 | +h0 upper outlet height (round-1 H1) | 11 | 0.208 / 0.239 / 0.238 / 0.267 | 0.2388 | 0.0120 | 0.405 | no |
| S07 | +FIR input taps (round-1 F1) | 12 | 0.137 / 0.251 / 0.176 / 0.247 | 0.2082 | 0.0263 | 0.354 | no |
| S08 | +cA1 upper-tank capacity slope | 11 | 0.152 / 0.246 / 0.235 / 0.237 | 0.2207 | 0.0195 | 0.372 | no |
| S09 | +tp +w3 | 12 | 0.187 / 0.221 / 0.208 / 0.246 | 0.2166 | 0.0124 | 0.328 | no |
| S10 | +tp +to | 12 | 0.148 / 0.252 / 0.164 / 0.285 | 0.2199 | 0.0327 | 0.351 | no |
| S11 | +qov +to | 12 | 0.139 / 0.252 / 0.177 / 0.247 | 0.2091 | 0.0261 | 0.397 | no |
| S12 | +tp +w3 +to | 13 | incomplete (deadline) | - | - | - | excluded |
| S13 | swap: cA replaced by w3 | 10 | 0.223 / 0.252 / 0.199 / 0.237 | 0.2287 | 0.0111 | 0.368 | no |

Rule 1: best pooled CV is S02 (0.1934, SE 0.0144), so the threshold is 0.2078; eligible are S01 (0.2030), S02 and S04 (0.2060), all 11 numbers; the tie on size is broken by the level-extrapolation RMSE (S02 0.338, S01 0.351, S04 0.396): **S02 = `ct_phys8` + w3** is chosen. The pooled CV improves from 0.2091 to 0.1934 (the two worst blocks, 1 and 3, from 0.252 and 0.247 to 0.225 and 0.185; blocks 0 and 2 get worse, 0.139 to 0.156 and 0.177 to 0.201), the level RMSE from 0.396 to 0.338, and the full-record training RMSE from 0.17848 to 0.16281 (dry run, seed 0, 8 starts x 600 iterations; parameters k1 0.1535, r2 0.9953, k3 0.1816, k4 0.2034, beta 0.7162, a 1.466, b4 0.986, cA -0.1268, w3 -0.1400).

How firm this is, in plain words:
- The choice is borderline. `ct_phys8` falls 0.0013 V outside the threshold, and the FIR candidate (S07, 0.2082) 0.0004 V outside. The threshold uses the standard error of the best candidate (S02: 0.0144), the round-1 convention; with `ct_phys8`'s own 0.026 it would have been eligible and, being smallest, would have been kept. Four folds give a noisy standard error.
- The `w3` family looks weakly identified. Across the five jobs S02's sensor offset `a` and pump offset `b4` range from -3.1 to 1.9 and from -1.6 to 1.2 (`ct_phys8`: 1.1 to 2.0 and 1.3 to 1.6), and in two of the five jobs a random start clearly beat the warm start at `ct_phys8`'s optimum. S09 (`tp` + `w3`, a superset of S02) has the lower training loss on four of five jobs but the worse held-out error (0.2166). Differences of 0.01-0.02 V between structures are within what four folds can resolve.
- Stage 2 never ran. The 100-minute job deadline had passed when rule 1 finished (the registration says "after the kernel's start", the code measured from the start of `select2.py`, which began 4 minutes after the kernel: 17:06:27 instead of 17:02:26 UTC; checked from `select.log` and the job files, the latest job started at 17:02:22, so no job fell into the 4-minute difference and the outcome is unchanged), because my per-job time estimate (100-140 s) was 2.5-3 times too low (345 s on average, four jobs sharing four vCPUs). By the registered rule the unevaluated fit variants are excluded and rule 2 keeps the baseline fit G0. More random starts, the pseudo-Huber loss and multiple shooting are therefore untested, not rejected. S12 (`tp` + `w3` + `to`) lost its fourth block and level job to the same deadline and is excluded.
- The estimation-record numbers say nothing yet about the test record: round 1 selected on a CV of 0.209 and scored 0.272.

### 9.3 Frozen run (kernel `sankalpsthakur/bench-control-ct-round2-final` v1, 2026-10-09 17:31-18:11 UTC, private)

Configuration (`results/FREEZE_ct_round2.json`, committed in 4968eec, generated mechanically from the selection kernel's `selection.json`): S02 = `ct_phys8` + the level-dependent lower-outlet coefficient `w3`; 9 free parameters (k1, r2, k3, k4, beta, a, b4, cA, w3) and the 2 initial states fitted per record from the first 50 samples; simulation-error MSE on the whole estimation record, L-BFGS-B on JAX float64 gradients, 8 random starts x 600 iterations (as in round 1), seeds 0-9. The ten train-only fits ran in a pool of four workers; the gate (train RMSE of seed 0 equal to the selection kernel's dry run, 0.16281, within 0.003) passed before the test record was touched; then each seed was scored once (TEST-LOG.md rows 25-34). The kernel verified the sha256 of all 49 payload files, the git tree hashes of `ct/` (120f81f0, the round-1 tree) and `ct2/` (ff0c9b38), the freeze blob (768aceb9) and the pre-registration blob, and that `ct2/phys2.py` and `ct2/select2.py` are the files the selection kernel had verified; data zip and `.mat` sha256 matched.

| seed | test RMSE [V] | train RMSE [V] | 95 % block-bootstrap interval of this seed's RMSE | fit time on a shared vCPU [s] |
|---|---|---|---|---|
| 0 | 0.2362 | 0.16281 | 0.1999 .. 0.2743 | 1002 |
| 1 | 0.2362 | 0.16281 | 0.1997 .. 0.2743 | 1080 |
| 2 | 0.2362 | 0.16281 | 0.1997 .. 0.2744 | 567 |
| 3 | 0.2363 | 0.16281 | 0.2000 .. 0.2744 | 912 |
| 4 | 0.2362 | 0.16281 | 0.1998 .. 0.2744 | 716 |
| 5 | 0.2362 | 0.16281 | 0.1997 .. 0.2743 | 669 |
| 6 | 0.2362 | 0.16281 | 0.1998 .. 0.2743 | 988 |
| 7 | 0.2362 | 0.16281 | 0.1997 .. 0.2744 | 911 |
| 8 | 0.2363 | 0.16281 | 0.1999 .. 0.2743 | 782 |
| 9 | 0.2362 | 0.16281 | 0.1998 .. 0.2743 | 616 |
| **mean** | **0.23621** (sd over seeds 0.00004) | 0.16281 | **0.1998 .. 0.2743** (interval of the mean statistic, same resampled blocks for every seed) | |

All ten seeds reach the same optimum (train RMSE 0.16281; a 1.46-1.47, b4 0.98-0.99, w3 -0.140), so the seed spread (4e-5 V) says nothing about data uncertainty; the interval is a moving-block bootstrap (block 25, 4000 draws) of the 974 scored test residuals, and it is wide for the same reason as in round 1.
Where the remaining error sits (read off `results/figs/ct_ct_round2_test.png`, approximate): the model is too low at the fast rises, by about 0.7 V at sample 140-150 before the first large peak and by about 0.5-0.6 V before and at the second large peak (samples 380-410), and 0.3-0.45 V too low between samples 520 and 620; the largest positive errors are about +0.5 V near samples 190 and 840. The error is spread over the record rather than sitting in one event.

Paired comparison with the 0.221 entry's code (11 seeds, rerun by us, `ct/compare_ref.py`, block 25, same resampled indices for both models; our seed 0 shown, the other nine differ by at most 0.0002 V):

| reference seed | its RMSE | ours minus reference | 95 % paired interval | upper bound below 0 for |
|---|---|---|---|---|
| 42 (notebook default; the 0.221 entry) | 0.2163 | +0.0199 | -0.0184 .. +0.0584 | 0 of 10 of our seeds |
| 3 | 0.2383 | -0.0021 | -0.0367 .. +0.0343 | 0 of 10 of our seeds |
| 8 | 0.2400 | -0.0038 | -0.0546 .. +0.0499 | 0 of 10 of our seeds |
| 9 | 0.2469 | -0.0108 | -0.0584 .. +0.0383 | 0 of 10 of our seeds |
| 0 | 0.2516 | -0.0154 | -0.0693 .. +0.0435 | 0 of 10 of our seeds |
| 7 | 0.2561 | -0.0199 | -0.0615 .. +0.0235 | 0 of 10 of our seeds |
| 1 | 0.2658 | -0.0296 | -0.0687 .. +0.0103 | 0 of 10 of our seeds |
| 5 | 0.2795 | -0.0433 | -0.1057 .. +0.0199 | 0 of 10 of our seeds |
| 2 | 0.3093 | -0.0731 | -0.1378 .. -0.0073 | 10 of 10 of our seeds |
| 4 | 0.3245 | -0.0883 | -0.1450 .. -0.0335 | 10 of 10 of our seeds |
| 6 | 0.3465 | -0.1103 | -0.1733 .. -0.0465 | 10 of 10 of our seeds |

Pre-registered claim rule (copied from round 1): "beats the 0.221 entry" needs the paired 95 % upper bound against the seed-42 trajectory below 0 for every one of our seeds. **Not met**: the difference to the 0.221 entry is +0.0199 V (95 % paired interval -0.0184 .. +0.0584), the upper bound is above 0 for all ten seeds, the mean 0.2362 V is above 0.221 V, and 20 % of the bootstrap draws of the mean fall below 0.221 V. Fraction of the other ten reference seeds beaten under the same paired rule: 0.3 (seeds 2, 4 and 6). So no "best published result" wording is allowed. What can be said: an 11-number grey-box model, with a candidate set chosen after the round-1 test-residual analysis (so not an independent test of the record), reaches 0.236 V (95 % block-bootstrap interval 0.200-0.274 V) on the official Cascaded Tanks split, free-run simulation, n_init = 50, self-evaluated on 2026-10-09; it does not beat the 0.221 V entry (the entry's own code scores 0.216-0.347 V over 11 seeds, mean 0.270; our point estimate is below 10 of those 11 seeds and above the notebook-default seed that produced the 0.221 figure), and the comparison is inconclusive in both directions.

Round 1 against round 2:

| | `ct_phys8` (round 1) | `ct_round2` (S02, round 2) |
|---|---|---|
| numbers (parameters + initial states) | 10 | 11 |
| pooled CV on the estimation record | 0.2091 | 0.1934 |
| level-extrapolation RMSE | 0.396 | 0.338 |
| training RMSE | 0.17848 | 0.16281 |
| **test RMSE, mean of 10 seeds** | **0.2719** | **0.2362** |
| 95 % block-bootstrap interval of the mean | 0.216 .. 0.339 | 0.200 .. 0.274 |
| bootstrap draws of the mean below 0.221 | 4 % | 20 % |
| paired difference to the 0.221 entry (seed 42) | +0.0557 (-0.0025 .. +0.1232) | +0.0199 (-0.0184 .. +0.0584) |
| other reference seeds beaten (upper bound below 0) | 0 of 10 | 3 of 10 |

### 9.4 Edge numbers (kernel `bench-control-ct-round2-final`, train record only; `results/ct_edge_ct_round2.json`)

- Size: 11 numbers (9 parameters + the 2 initial states of a record; at deployment the initial state is re-fitted from the first 50 samples). Parameter payload 44 B in fp32, 22 B in fp16, 15 B in int8 (11 values + a 4-byte scale); as `.npy`/`.npz` containers 172 / 150 / 513 B. The 0.221 entry's code has at least 330 parameters.
- Quantisation, effect on the training-record RMSE (float64: 0.16281 V): fp32 -8e-12 V; fp16 0.16293 V (+1.2e-4 V, largest trajectory deviation 0.015 V); **int8 with one shared scale fails**: 1.236 V (+1.07 V; parameter errors up to 14 %). fp16 is safe, int8 per tensor is not (as in round 1).
- C99 step and its time: not applicable. `ct/edge.py`'s emitter covers the terms of `ct/phys.py` except `h0` and none of the round-2 terms; `w3` adds one `exp` per right-hand-side evaluation and would need an emitter extension, which I did not write. Round 1's C99 step of `ct_phys8` (289 ns per sample on a Kaggle Xeon, equal to JAX within 5e-13 V) is the nearest number; round 2 has none. No M4 latency (the machine was loaded).

### 9.5 Reading, and what is left open

- The point estimate improved by 0.036 V (13 %) and the direction agrees with the estimation-record evidence (pooled CV 0.209 to 0.193, level RMSE 0.396 to 0.338, training RMSE 0.178 to 0.163). It remains above 0.221 V, the two test intervals overlap almost completely, and the pre-registered rule is not met. I did not compute a paired comparison of the round-1 and round-2 predictions (the predictions stay in the kernel output; only json/log/png were fetched), so "better than `ct_phys8` on the test record" is a point estimate, not a tested statement.
- The improvement cannot be separated from the test-informed choice of candidate families (section 9.1), from the borderline selection (`ct_phys8` 0.0013 V outside the threshold, a different but defensible standard-error convention would have kept it), or from the multiplicity of 13 compared structures on four folds. A board resubmission of this number would also have to clear nonlinearbenchmark.org's rule against using test data to tune structure; I would not send it without the lead's decision on that.
- Open items, not rejected ones: the three fit variants (more random starts, pseudo-Huber loss, multiple shooting) for S02, and S12 (`tp` + `w3` + `to`), never ran for lack of time; a rerun needs a job deadline sized to about 350 s per job (the full registered plan is about 2.5 hours of four-worker time). `w3`-type terms look weakly identified in the fold fits (section 9.2), although the ten full-record fits agree.

### 9.6 Reproduction

- Registration and amendments: `PREREGISTRATION-CT-ROUND2.md` (commit 0509f85, amendments 1 and 2 in 15dcce8 and e0cf67f, outcome record in 4968eec). Code: `ct2/` (`phys2.py`, `select2.py`, `final2.py`, `make_freeze.py`, `selftest.py`); `tools/ct2_edge_run.py`, `tools/ct2_ingest_testlog.py`, `tools/ct2_tables.py`, `tools/ct2_report_tables.py`.
- Selection: `kernels/ct_round2_select/make_kernel.py` builds the kernel script from the committed sources (sha256 manifest inside); outputs in `logs/kaggle-output/ct-round2-select/` (`out/select/selection.json`, 85 job files, `select.log`). Freeze: `python ct2/make_freeze.py <selection.json> "<kernel ref>" results/FREEZE_ct_round2.json`.
- Final: `kernels/ct_round2_final/make_kernel.py`; outputs in `logs/kaggle-output/ct-round2-final/` and `results/ct_*_ct_round2.json`, `results/figs/ct_ct_round2_*.png`; test rows in TEST-LOG.md (25-34). Local fit-free checks: `bin/guard.sh -- .venv/bin/python ct2/selftest.py <name>` (equivalence to `ct/phys.py` to 2e-15 V, limits, gradients, warm starts, rules).
