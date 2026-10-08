# Answers for the nonlinearbenchmark.org result submission form

Form: https://forms.gle/1FNGkpZ5X7kf6ibw7 (Google Form "Result Submission Form - Nonlinearbenchmark.org"). Rules: https://www.nonlinearbenchmark.org/results-reporting

| field | value |
|---|---|
| Email address | the author's address |
| Author names | Sankalp Thakur |
| Method name | Grey-box two-tank Torricelli ODE with overflow routing, 8 physical parameters (ct_phys8) |
| Link to paper/technical report | https://github.com/sankalpsthakur/cascaded-tanks-phys8/blob/main/REPORT.md |
| Link to Git code repository | https://github.com/sankalpsthakur/cascaded-tanks-phys8 |
| Git Commit ID / SHA / Hash | <commit SHA after push> |
| Benchmark System | Cascaded Tanks |
| RMSE test results | 0.272 V (mean of 10 optimiser seeds, range 0.2717-0.2720; official RMSE(y, y_model, n_init=50), free-run simulation; block-bootstrap 95 % interval of the mean 0.216-0.339 V) |
| Approximate training time | 1-60 min (232-294 s per fit on one Kaggle vCPU; about 2 min on an Apple M4) |
| Training platform | Kaggle CPU container, Linux x86_64: Intel Xeon @ 2.20 GHz, 4 vCPU (1 thread used), 31 GB RAM, no GPU; JAX 0.11.2 float64 CPU, SciPy 1.18.1 L-BFGS-B |
| White- to black-box (1-10) | 2 |
| Author remarks | see below |

Author remarks:

Physics-structured model: two cascaded Torricelli tanks with overflow routing, a bottle-shaped lower tank (capacity slope), a signed pump offset and a sensor offset; 8 free physical parameters, no neural or residual part; the two initial states are fitted per record from the first 50 samples only (the official state-initialisation window). Structure chosen by blocked cross-validation on the estimation record only; the test record was not used for tuning or selection; the configuration was frozen and committed before the first score of this model. Ten optimiser seeds converge to the same optimum (test RMSE 0.2717-0.2720), so the seed spread carries no information about data uncertainty; a moving-block bootstrap of the 974 scored samples gives a 95 % interval of 0.216-0.339 V. This does not beat the 0.221 V entry; a paired block-bootstrap comparison against that entry's own code (11 seeds, rerun by us) is inconclusive in both directions. Prior art with a similar model class on the pre-official split: Rogers, Holmes, Cross, Worden 2017 (grey-box with physical overflow model, 0.18 V, listed as legacy). Loader note: nonlinear_benchmarks 1.0.1's default 4TU URL returns an HTML page; we passed its Google-Drive fallback URL.
