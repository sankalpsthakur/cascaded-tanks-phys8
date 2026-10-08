# TEST-LOG

Every evaluation that touches the official Cascaded Tanks TEST record is appended here, including
reproduction runs of third-party code and failed runs. Nothing in this table was used for selection
unless the `purpose` column says so explicitly (it should never).

RMSE column = official `nonlinear_benchmarks` metric: free-run simulation, first 50 test samples are the
state-initialisation window and excluded (n_init=50), volts.

| # | UTC | tag | config | seed | RMSE n_init=50 | extra | purpose | code @ |
|---|-----|-----|--------|------|----------------|-------|---------|--------|
| 1 | 2026-10-07 03:11:22 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 42 | 0.2163 | repo-convention test[6:] RMSE=0.2186; wall=63s | reproduction of published entry (metric check); not used for selection | nogit |
| 2 | 2026-10-07 03:11:50 | REPRO third-party 0.221 code - crashed attempt 1 | 44c15c884c | 42 | n/a (crash) | DataLoader worker died (macOS spawn); no trajectory scored; physical-fit phase of the repo partially ran with test as its val set | failed reproduction attempt, logged for completeness | nogit |
| 3 | 2026-10-07 03:11:50 | REPRO third-party 0.221 code - crashed attempt 2 | 21416370d2 | 42 | n/a (crash) | display_step=1000 -> val phase never ran -> UnboundLocalError after 100 neural epochs; the repo printed physical-phase x0/theta but no RMSE was read | failed reproduction attempt, logged for completeness | nogit |
| 4 | 2026-10-07 03:12:52 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 0 | 0.2516 | repo-convention test[6:] RMSE=0.2549; wall=66s | reproduction of published entry (metric check); not used for selection | nogit |
| 5 | 2026-10-07 03:14:07 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 1 | 0.2658 | repo-convention test[6:] RMSE=0.2662; wall=69s | reproduction of published entry (metric check); not used for selection | nogit |
| 6 | 2026-10-07 03:15:24 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 2 | 0.3093 | repo-convention test[6:] RMSE=0.3102; wall=72s | reproduction of published entry (metric check); not used for selection | nogit |
| 7 | 2026-10-07 03:16:36 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 3 | 0.2383 | repo-convention test[6:] RMSE=0.2402; wall=66s | reproduction of published entry (metric check); not used for selection | nogit |
| 8 | 2026-10-07 03:18:18 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 4 | 0.3245 | repo-convention test[6:] RMSE=0.3282; wall=97s | reproduction of published entry (metric check); not used for selection | nogit |
| 9 | 2026-10-07 03:19:37 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 5 | 0.2795 | repo-convention test[6:] RMSE=0.2783; wall=73s | reproduction of published entry (metric check); not used for selection | nogit |
| 10 | 2026-10-07 03:20:49 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 6 | 0.3465 | repo-convention test[6:] RMSE=0.3472; wall=66s | reproduction of published entry (metric check); not used for selection | be398db+dirty |
| 11 | 2026-10-07 03:21:57 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 7 | 0.2561 | repo-convention test[6:] RMSE=0.2564; wall=62s | reproduction of published entry (metric check); not used for selection | be398db+dirty |
| 12 | 2026-10-07 03:23:07 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 8 | 0.2400 | repo-convention test[6:] RMSE=0.2415; wall=65s | reproduction of published entry (metric check); not used for selection | be398db+dirty |
| 13 | 2026-10-07 03:24:22 | REPRO third-party 0.221 code (Main.ipynb defaults) | f4d5ea47d6 | 9 | 0.2469 | repo-convention test[6:] RMSE=0.2480; wall=70s | reproduction of published entry (metric check); not used for selection | be398db+dirty |
| 14 | 2026-10-07 04:22:32 | test record descriptive statistics viewed at load (min/max y, first 8 samples, saturation count); also the benchmark PDF figure | b6c19da80e | - | n/a | not a model evaluation; the test minimum 2.16 V (< estimation minimum 2.91 V) motivated the level-extrapolation validation on the estimation record | disclosure of test-record inspection before the freeze | 374934f+dirty |
| 15 | 2026-10-07 10:23:09 | FROZEN ct_phys8 | 3d084784f5 | 0 | 0.2720 | train_rmse=0.1785; fit=294s | frozen-config final run (one score per seed) | 62034c2 |
| 16 | 2026-10-07 10:27:19 | FROZEN ct_phys8 | 3d084784f5 | 1 | 0.2719 | train_rmse=0.1785; fit=251s | frozen-config final run (one score per seed) | 62034c2 |
| 17 | 2026-10-07 10:31:36 | FROZEN ct_phys8 | 3d084784f5 | 2 | 0.2720 | train_rmse=0.1785; fit=256s | frozen-config final run (one score per seed) | 62034c2 |
| 18 | 2026-10-07 10:36:10 | FROZEN ct_phys8 | 3d084784f5 | 3 | 0.2719 | train_rmse=0.1785; fit=273s | frozen-config final run (one score per seed) | 62034c2 |
| 19 | 2026-10-07 10:40:02 | FROZEN ct_phys8 | 3d084784f5 | 4 | 0.2717 | train_rmse=0.1785; fit=232s | frozen-config final run (one score per seed) | 62034c2 |
| 20 | 2026-10-07 10:44:18 | FROZEN ct_phys8 | 3d084784f5 | 5 | 0.2718 | train_rmse=0.1785; fit=256s | frozen-config final run (one score per seed) | 62034c2 |
| 21 | 2026-10-07 10:48:19 | FROZEN ct_phys8 | 3d084784f5 | 6 | 0.2718 | train_rmse=0.1785; fit=241s | frozen-config final run (one score per seed) | 62034c2 |
| 22 | 2026-10-07 10:53:03 | FROZEN ct_phys8 | 3d084784f5 | 7 | 0.2719 | train_rmse=0.1785; fit=284s | frozen-config final run (one score per seed) | 62034c2 |
| 23 | 2026-10-07 10:57:57 | FROZEN ct_phys8 | 3d084784f5 | 8 | 0.2718 | train_rmse=0.1785; fit=293s | frozen-config final run (one score per seed) | 62034c2 |
| 24 | 2026-10-07 11:02:04 | FROZEN ct_phys8 | 3d084784f5 | 9 | 0.2720 | train_rmse=0.1785; fit=247s | frozen-config final run (one score per seed) | 62034c2 |
