# Pre-registration: Cascaded Tanks, round 2 (`ct_round2`)

Written 2026-10-09 15:20 UTC by the control builder (Sankalp Thakur's free-compute benchmark program), BEFORE any round-2 fitting run.
Round 1 (`ct_phys8`, frozen 2026-10-07) scored a mean test RMSE of 0.2719 V (10 seeds; 95 % block-bootstrap interval of the mean 0.216-0.339 V); the board's best official-split entry is 0.221 V.
This file fixes the candidate set, the selection rule, the freeze and the claim rule for one more shot. It is committed together with the code that implements it (`ct2/`); any later change is an amendment in section 12, dated, and made before the selection kernel is pushed.

## 1. What is fixed

- `ct/` and `results/FREEZE_ct_phys8.json` stay byte-identical to round 1 (git tree hash of `ct/`: `120f81f08ff5460b0a8d79758ebc28364ba9eca7`; the kernels recompute and check it). Round-2 code lives in `ct2/`: `phys2.py` (simulator and fitter), `select2.py` (selection driver), `final2.py` (frozen run), `make_freeze.py` (selection.json to freeze file), `selftest.py` (fit-free checks). The kernels check the git tree hash of `ct2/` and a sha256 manifest of every payload file.
- Selection uses the ESTIMATION record only. `select2.py` replaces `ct.protocol.test_eval` by a function that raises, and the loader by a wrapper that returns `(train, None)`, so the selection code cannot read or score the test record. The selection kernel never calls the test scorer.
- Data: `nonlinear_benchmarks` 1.0.1 through the Google-Drive fallback URL of `ct/protocol.py`, zip and `.mat` sha256 checked in the kernel (as in round 1: `eb0fa058...51b`, `cb2f88d4...eab`). CC BY-SA 4.0.

## 2. Disclosure: what the test record already told us

Round 2 is NOT an independent test.
- Round 1 scored `ct_phys8` on the test record (TEST-LOG.md rows 15-24) and analysed the residuals: 86 % of the squared test error sits on the fast rise to the second large peak and its under-shoot (largest error -1.00 V at sample 386); the test minimum (2.16 V) is below the estimation minimum (2.91 V). Before round 1 the reference code of the 0.221 entry was run on the test record (rows 1-13) and summary statistics of the test record were printed once (row 14).
- The candidate families below (pump lag, level-dependent lower outlet, delayed overflow return, flow-dependent routing, upper outlet height, input FIR) were chosen AFTER that analysis, from physics and from the lead's list. The selection rule itself touches only the estimation record. So a round-2 score below 0.221 V would be a second look at the same record with a test-informed candidate set; it must be reported as such. nonlinearbenchmark.org's reporting rules forbid using test data to tune structure: whether any round-2 result may be sent there is the lead's decision, not this file's.
- In round 2 no code of ours reads test values before the freeze, and the saved round-1 test predictions (`runs/ct_final_ct_phys8_seed*_ypred.npy`) were not opened.
- Before this commit the only runs of round-2 code were fit-free checks and two-iteration plumbing runs (`ct2/selftest.py`, and the driver run in-process with 0.5 % of the iteration counts, results discarded). Their printed numbers were not used for any decision.
- Multiplicity: 14 structures and 3 fit variants are compared; the one-standard-error rule is the guard against picking the winner of a noisy comparison, and it is as noisy as four folds allow.

## 3. Model family

Time unit one sample (Ts = 4 s); levels in sensor volts; RK4 with 4 sub-steps per sample (h = 1/4); zero-order-hold input u; eps = 1e-4. This is `ct_phys8` with optional terms; terms not listed as free are absent (neutral).

```
q1  = k1 sqrt( max(x1 - h0, 0) + eps )                     h0 = 0 unless free
in  = k4 (ut - b4)  [ + c4 (ut - b4)^2 if c4 free ]         ut = input seen by the tank (pump lag, FIR: below)
dx1 = (in - q1) / max(1 + cA1 (x1 - 5), 0.2)                cA1 = 0 unless free
q3  = k3 sqrt(x2 + eps) exp( w3 (x2 - 5) / 5 )              w3 = 0 unless free
dx2 = (r2 q1 - q3) / max(1 + cA (x2 - 5), 0.2)
after every sub-step:  s = max(x1 - 10, 0);  x1 <- x1 - s;  routed = beta s     (qov free: beta h ((s/h + 1e-6)^qov - 1e-6^qov))
                       x2 <- x2 + routed;  x2 <- min(x2, 10 - a);  x1, x2 >= 0
y   = min(a + x2, 10)
```
- `tp` (pump/inflow lag, samples): x_p' = (u - x_p)/tp, x_p(0) = u_{-1} (steady state, no fitted number); per sub-step the tank sees the exact mean of the lagged input, ut = u + (x_p - u)(tp/h)(1 - exp(-h/tp)), then x_p <- u + (x_p - u) exp(-h/tp).
- `to` (delayed return of the routed overflow, samples): buffer b (b(0) = 0); b <- b + routed; delivered = b (1 - exp(-h/to)); b <- b - delivered; x2 <- x2 + delivered. `to -> 0` is `ct_phys8`.
- `w3`: level-dependent lower-outlet coefficient (one number; `w3 = 0` is `ct_phys8`).
- `qov`, `c4`, `h0`, `cA1`, FIR taps (`d1`, `d2`: ut = w0 u_t + w1 u_{t-1} + w2 u_{t-2}, w = softmax(0, d1, d2)) are the terms of `ct/phys.py` (round-1 definitions; `ct2/selftest.py` checks that `ct2/phys2.py` reproduces `ct/phys.py` to 2e-15 V for C, FIR, h0, qov, c4, cA1, free exponents and the sensor curvature).
- Positive parameters (k1, r2, k3, k4, beta, tp, to, qov) are fitted as logarithms; the others as real numbers. The two initial states (log x1, log x2) are fitted per record from the first 50 samples only, with all other parameters frozen (window fit of `ct/phys.py`, unchanged). "Numbers" of a structure = free parameters + 2 initial states.

## 4. Candidates (stage 1)

Base `B` = (k1, r2, k3, k4, beta, a, b4, cA) = `ct_phys8` (8 + 2 = 10 numbers).

| id | structure | numbers | comment |
|---|---|---|---|
| S00 | B | 10 | `ct_phys8`, the reference; must reproduce round-1 CV 0.2089 (gate, section 6) |
| S01 | B + tp | 11 | first-order pump / inflow lag |
| S02 | B + w3 | 11 | level-dependent lower-tank outlet coefficient |
| S03 | B + to | 11 | delayed (partial) return of the upper-tank overflow |
| S04 | B + qov | 11 | flow-dependent routing of the overflow (round-1 O1, never evaluated) |
| S05 | B + c4 | 11 | pump curvature (round-1 Q1, never evaluated) |
| S06 | B + h0 | 11 | upper outlet height (round-1 H1: 0.2336) |
| S07 | B + d1 + d2 | 12 | input FIR taps (round-1 F1: 0.2084) |
| S08 | B + cA1 | 11 | upper-tank capacity slope (round-1 Q2, never evaluated) |
| S09 | B + tp + w3 | 12 | |
| S10 | B + tp + to | 12 | pump lag and delayed overflow |
| S11 | B + qov + to | 12 | overflow routing, flow-dependent and delayed |
| S12 | B + tp + w3 + to | 13 | |
| S13 | B - cA + w3 | 10 | swap of the capacity slope for the outlet coefficient (same size as C) |

All candidates are at most 13 numbers. Fit variants (stage 2, applied to the structure that rule 1 selects):

| id | fit | CV starts | final starts |
|---|---|---|---|
| G0 | MSE, single shooting (round 1) | 4 | 8 |
| G3 | MSE, single shooting, more random starts | 8 | 16 |
| G1 | pseudo-Huber loss, 2 d^2 (sqrt(1 + (e/d)^2) - 1), d = 0.2 V (equals MSE for small errors, linear tails) | 4 | 8 |
| G2 | multiple shooting: 8 segments of 128 samples, each with its own free initial state, loss = MSE + 30 sum(gap^2) / 1024 over the 7 boundaries (gap = end state of a segment minus start state of the next, in volts; 30 = sample-equivalents per squared volt). Lag / buffer states restart at their block-start values in every segment | 4 | 8 |

## 5. Fitting

L-BFGS-B (scipy) on JAX float64 gradients, maxiter 400 and maxfun 800 in selection, maxiter 600 in the final fit; random starts as in `ct/phys.py` (numpy `default_rng(seed)`, positive parameters log(default) + N(0, 0.35^2), real parameters default + N(0, scale^2), initial states from the first output), best start by objective. CV fold b uses seed 100 b; the level check seed 0.
Warm start (fairness): every candidate nested in C (S01-S12) gets ONE extra start at C's optimum of the same job (same fold / same level mask, from the S00 run in the same kernel) with the new terms switched off (tp = to = 0.02 samples, w3 = 0, qov = 1, c4 = cA1 = h0 = 0, d1 = d2 = -12; the loss at that start is within 0.2 % of C's, `ct2/selftest.py warm-<id>`). So the training loss of a richer fit cannot end above C's because of an unlucky start. S13 is not nested and gets none. In stage 2 every variant gets one extra start at the G0 optimum of the same job. (Multiple shooting converts a start into its segment states by simulating once.)

## 6. Selection protocol (all on the ESTIMATION record)

- Blocked CV, as in round 1: blocks [0,256), [256,512), [512,768), [768,1024). For each block the model is fitted on the whole record with that block's loss weight set to 0 (the state runs through the block driven by u), the block's initial state is fitted from its first 50 samples with the parameters frozen, the block is free-run and scored by RMSE over samples 50..255. Pooled CV RMSE = sqrt(mean of the four block MSEs).
- Standard error of a candidate's pooled CV RMSE (delta method on the four block MSEs, which reproduces round 1's "about 0.025": 0.026 for C): `SE = sd_ddof1(block MSEs) / sqrt(4) / (2 * pooled)`.
- Level-extrapolation check, as in round 1: samples with y below the 12 % quantile of the estimation outputs (y < 3.333 V, 123 samples; the first 50 samples are never masked) get loss weight 0, 6 starts, seed 0; the score is the RMSE on the masked samples of the free run from the fitted initial state of the whole record.
- Jobs: per candidate and fit variant 5 jobs (4 folds + level), run in a pool of 4 single-thread worker processes in a private Kaggle CPU kernel. A candidate is complete only if its 5 jobs finished with finite numbers. Jobs not started 100 minutes after the kernel's start are skipped and the candidate is excluded and listed as incomplete.
- Reproduction gate: wave 1 runs S00 and S13. If S00's pooled CV differs from round 1's 0.2089 by more than 0.003 the kernel stops (nothing else is run) and I report that and debug; S00's level RMSE (round 1: 0.396) and folds (0.139, 0.252, 0.177, 0.246) are recorded.
- **Rule 1** (structure, fit G0): best = lowest pooled CV among complete candidates; eligible = pooled CV <= best + SE(best); chosen = the eligible candidate with the fewest numbers; among candidates with the same number of numbers the lowest level-extrapolation RMSE; level RMSEs closer than 0.005 V count as equal and the lower candidate index wins.
- **Rule 2** (fit variant on the structure chosen by rule 1; G0 comes from stage 1): same arithmetic among {G0, G3, G1, G2} (all the same size): eligible within one SE of the best pooled CV, then the lowest level-extrapolation RMSE; level RMSEs within 0.005 V count as equal and the order G0 > G3 > G1 > G2 decides.
- A larger structure replaces C only if C falls outside `best + SE(best)`, i.e. with SE about 0.025 only for a pooled CV of roughly 0.18 or lower. The most likely outcome of rule 1 is therefore C (S00) or the same-size swap S13; the stage-2 fit variants are then the real choice. This is registered here so that a re-selection of C is read as a result, not as a failure.

## 7. Dry run, identity test, branches

- Dry run (TRAIN only, in the selection kernel): the chosen structure with the chosen fit variant and its final hyper-parameters (final starts above, 600 iterations, seed 0) is fitted on the full estimation record. It gives the gate value (train RMSE of the free run from the fitted initial state) for the final kernel and the identity test.
- Identity test (only when rule 1 chose S00 and rule 2 chose G0 or G3): the configuration is "identical to `ct_phys8`" if the dry-run parameters equal round 1's seed-0 parameters (`results/ct_final_ct_phys8.json`) to a relative difference below 1e-3 for all 8 and the train RMSE differs by less than 5e-4 V.
- **Branch A (identical):** `results/FREEZE_ct_round2.json` records `identical_to_ct_phys8: true` and `ct2/final2.py` refuses to run; no test record is touched; TEST-LOG.md rows 15-24 are the scores. Round 2 then reports: the pre-registered selection reproduced `ct_phys8`; 0.2719 V (0.216-0.339 V) stands; second negative by construction.
- **Branch B (anything else):** freeze (section 8), then the final kernel.

## 8. Freeze and final run

- `ct2/make_freeze.py` writes `results/FREEZE_ct_round2.json` from the selection kernel's `selection.json` mechanically (structure, fit, hyper-parameters, the selection tables, the gate value, seeds 0-9, the claim rule); I commit it before the final kernel is built. Nothing is edited by hand.
- Final run (`bench-control-ct-round2-final`): seeds 0-9; the ten train-only fits run in a 4-worker pool; the train RMSE of seed 0 must equal the dry run's within 0.003 V, otherwise the test record is not touched; then each seed is scored ONCE on the test record in the parent process (`ct.protocol.test_eval`), one row per scoring in `TEST-LOG-CT2.md` (rows numbered from 25; ingested into TEST-LOG.md), failures included. The test record is touched once per frozen configuration. No seed is re-run, no variant is added after the freeze.
- Then `ct/compare_ref.py ct_round2` (paired moving-block bootstrap, block 25, 4000 draws, against the 11 reference-code trajectories of the 0.221 entry), `ct/plots.py`, `tools/ct_aggregate.py` (mean over seeds, block-bootstrap interval of the mean), and the train-only edge numbers.
- Claim rule: identical to round 1 (`claim_rule` copied verbatim into the freeze; only the date in the wording is the run's). Headline = mean over seeds 0-9, never the best seed; uncertainty = block bootstrap of the test residuals, not the seed spread; "best published result on Cascaded Tanks, official split, n_init=50 free-run simulation RMSE, self-evaluated, as of 2026-10-09" is allowed only if, for every seed, the 95 % upper bound of the paired difference to the reference code's seed-42 trajectory is below 0, and then only together with the disclosure of section 2. A result at or above 0.221 V is reported as-is. Never "#1" or "SOTA".

## 9. Edge numbers

Parameter count (numbers), bytes in fp32 / fp16 / int8 (with scale) of the parameter vector, the effect of fp16 and int8-per-tensor quantisation on the TRAIN RMSE (JAX float64 simulator), and, where the selected structure is expressible by `ct/edge.py`'s C99 emitter (terms of `ct/phys.py` except h0), the C99 step time and its agreement with JAX on a train input, in the kernel on the Kaggle Xeon. For `tp`, `to`, `w3` or `h0` the emitter does not apply; I report that instead of extending it unless time allows. Branch A: round 1's numbers stand.

## 10. Infrastructure

Private Kaggle CPU kernels `bench-control-ct-round2-select` and `bench-control-ct-round2-final` (names fixed by the lead), built like `kernels/ct_final/make_kernel.py` (sha256 manifest of every payload file, tree hashes of `ct/` and `ct2/`, data from the Drive URL with checksum, train-only gate). One CPU slot under `locks/kaggle-cpu-N` with `owner` = "control <kernel ref> <UTC> expected end ...", one background waiter, only json/log/png outputs fetched, lock removed at the end. Local python only through `bin/guard.sh` (60 s, 512 MiB); no fitting on the Mac. Time box 5 hours from 14:50 UTC; a kernel still running at the end keeps its lock with the expected end and is noted in STATUS.md.

## 11. Not done, on purpose

No residual neural component (round 1: worse), no dead-zone pump (worse), no free exponents (overfits), no sensor curvature (unstable window fit), no bagging (worse), no change of the 50-sample window protocol, no use of the test record for anything before the freeze.

## 12. Amendments

- 2026-10-09 15:24 UTC, after the selection kernel was pushed (15:22 UTC, `ct2/` tree 154418ac), before any result exists: `ct2/final2.py` got an in-process option (`--workers 0`) and the fit workers load the train record through `load_train()`; `ct2/selftest.py` got a `final_dummy` test (final2.py end to end with the TRAIN record standing in for the test record, so no test value is read). `ct2/phys2.py` and `ct2/select2.py` are unchanged (the selection kernel imports neither final2.py nor selftest.py's new test). The final kernel will carry the new `ct2/` tree hash and will check that the sha256 of `phys2.py` and `select2.py` equal the ones the selection kernel verified.
- 2026-10-09 15:30 UTC, selection kernel still running, no result exists: `ct2/final2.py` additionally stores the fitted vector `z` of every seed in `results/ct_final_ct_round2.json` (the edge stage then needs no refit). Added `tools/ct2_edge_run.py`, `tools/ct2_ingest_testlog.py` and `kernels/ct_round2_final/` (builder, main). None of these takes part in the selection; `ct2/phys2.py` and `ct2/select2.py` are still byte-identical to the selection kernel's.
- Record (not an amendment; no rule, candidate or number changed), 2026-10-09 17:30 UTC. The selection kernel `bench-control-ct-round2-select` v1 ran `select2.py` from 15:26 to 17:16 UTC. Reproduction gate passed (S00 pooled CV 0.2091 against round 1's 0.2089, folds 0.139 0.252 0.177 0.247, level RMSE 0.396 against 0.396). Stage 1 completed 13 of the 14 candidates; S12 lost its fourth block and its level job to the 100-minute deadline and is excluded, as registered. The deadline (100 minutes after the start of `select2.py`) had passed when rule 1 finished, so no stage-2 job (G3, G1, G2) started: my per-job time estimate (100-140 s) was 2.5-3 times too low (345 s on average with four jobs sharing four vCPUs). By the registered rule the unevaluated variants are excluded and rule 2 keeps G0. The fit variants (more random starts, pseudo-Huber, multiple shooting) are therefore NOT evaluated in round 2; they are an open item, not a rejected one. Rule 1 chose S02 (B + w3, 11 numbers): best pooled CV 0.1934, SE 0.0144, threshold 0.2078; eligible S01 (0.2030), S02, S04 (0.2060); C (0.2091) falls outside by 0.0013; ties on size resolved by the level RMSE (S02 0.338, S01 0.351, S04 0.396). The dry run (S02 + G0, 8 starts x 600 iterations, seed 0, train only) gave train RMSE 0.16281.
- Record, 2026-10-09 18:20 UTC (checked after the final kernel; no rule changed): section 6 says jobs not started 100 minutes "after the kernel's start" are skipped; `select2.py` applied the 100 minutes from its own start (15:26:27 UTC), 4 minutes after the kernel's (15:22:26 UTC), i.e. 17:06:27 instead of 17:02:26 UTC. From `select.log` and the job files the latest job start was 17:02:22 UTC, so no job started in the 4-minute difference; the set of completed jobs, and the outcome, are the same under either reading.
