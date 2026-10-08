"""Reproduce the 0.221 V sheet entry: 'Normalized Gray-box State Space Neural Networks'
(Banitalebi Dehkordi, Forgione, Cimini, Frascati, Piga; Automatica 2025), code
github.com/MiladBnit/GreyBox-Neural-State-Space-Normalization @ 5f975bad.

Runs the repo's Main.ipynb configuration verbatim (augmentation='Exact - MANAM', 10 physical + 100 neural
epochs) as a script, then scores the produced test trajectory under (a) the repo's own convention
(test[6:], all 1018 samples) and (b) the official nonlinear_benchmarks protocol (n_init=50).

NOTE: the repo uses the TEST record as its 'validation' set and starts the test simulation from the
*training* initial-state estimate; this script reproduces that as-is, it does not fix it. Every run is
logged in TEST-LOG.md as a third-party reproduction (never used for selection of our own models).
"""
import argparse, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "third_party", "GreyBox-Neural-State-Space-Normalization")
sys.path.insert(0, ROOT)
from ct import protocol as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--augmentation", default="Exact - MANAM")
ap.add_argument("--epochs_phys", type=int, default=10)
ap.add_argument("--epochs_neural", type=int, default=100)
ap.add_argument("--threads", type=int, default=2)
args = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(args.threads)
os.chdir(REF)
sys.path.insert(0, REF)
from torch import nn  # noqa: E402
from Launcher import launch_greyboxmodel  # noqa: E402

neuron_dict = {"f_b": [8, 8], "g_b": [8, 8], "h_b": [8]}
act_fn_dict = {"f_b": nn.Tanh(), "g_b": nn.Tanh(), "h_b": nn.Tanh()}
optim_alg_dict = {"physical optimization algorithm": torch.optim.AdamW, "neural optimization algorithm": torch.optim.AdamW}
lr_neuralnet = {"f_b": 1e-2, "g_b": 5e-2, "h_b": 1e-2}
t0 = time.time()
y_pred_test, _, augmodel, result_dict, _, model, exec_time = launch_greyboxmodel(
    epochs_neural=args.epochs_neural, epochs_phys=args.epochs_phys, lr_neuralnet=lr_neuralnet,
    lr_theta_f=1e-2, lr_x0_phys=1e-2, lr_mu_x=1e-3, lr_sigma_x=1e-3, augmentation=args.augmentation,
    x_b_exploration_scale=100, init_std=1e-2, neuron_dict=neuron_dict, optim_alg_dict=optim_alg_dict,
    act_fn_dict=act_fn_dict, display_step=1, device=torch.device("cpu"), seed=args.seed,
    iv_est_dict={"Physical Estimate": {}}, gamma_power=0.1, dataloader_num_workers=0,
)
wall = time.time() - t0
yp = np.asarray(y_pred_test.detach().cpu().numpy(), float).reshape(-1)  # len 1018 = test[6:]
os.chdir(ROOT)
_, test = P._load_both()
y_full = np.asarray(test.y, float).ravel()
y6 = y_full[6:]
assert yp.shape == y6.shape, (yp.shape, y6.shape)
rmse_ref_conv = float(np.sqrt(np.mean((yp - y6) ** 2)))  # repo convention: whole test[6:]
# official protocol alignment: official window is test.y[50:], i.e. yp[44:]
y_full_model = np.concatenate([np.full(6, np.nan), yp])
rmse_off = float(np.sqrt(np.mean((y_full_model[50:] - y_full[50:]) ** 2)))
os.makedirs("runs", exist_ok=True)
np.savez(f"runs/ref_seed{args.seed}.npz", y_pred=yp, y_true6=y6)
res = dict(seed=args.seed, augmentation=args.augmentation, epochs_phys=args.epochs_phys, epochs_neural=args.epochs_neural,
           rmse_repo_convention_test6=rmse_ref_conv, rmse_official_ninit50=rmse_off, wall_s=wall, torch=torch.__version__)
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/ref_repro_seed{args.seed}.json", "w"), indent=1)
P.log_external(
    tag="REPRO third-party 0.221 code (Main.ipynb defaults)", config=dict(augmentation=args.augmentation, ep_phys=args.epochs_phys, ep_neural=args.epochs_neural),
    seed=args.seed, rmse=f"{rmse_off:.4f}",
    extra=f"repo-convention test[6:] RMSE={rmse_ref_conv:.4f}; wall={wall:.0f}s",
    purpose="reproduction of published entry (metric check); not used for selection")
print(json.dumps(res, indent=1))
