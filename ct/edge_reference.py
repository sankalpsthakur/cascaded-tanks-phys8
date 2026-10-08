"""Size/latency of the 0.221-entry model (grey-box SSNN, Main.ipynb config) WITHOUT training or touching the test record:
instantiates its neural components with the notebook's architecture and counts parameters; times one forward step of the
components (f_b, g_b, h_b + the physical f) at 1 thread, batch 1 (a lower bound for its per-sample latency: the repo's
simulator adds normalisation/projection ops and python loops)."""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, "third_party", "GreyBox-Neural-State-Space-Normalization")
sys.path.insert(0, REF)
import torch  # noqa: E402
torch.set_num_threads(1)
from torch import nn  # noqa: E402
import Classes as C  # noqa: E402

n_x, n_x_b, n_u, n_y = 2, 3, 1, 1
dev = torch.device("cpu")
bias = {"f_b": {"input layer": False, "hidden layer": False, "output layer": False},
        "g_b": {"input layer": False, "hidden layer": False, "output layer": False},
        "h_b": {"input layer": True, "hidden layer": True, "output layer": True}}
f_b = C.FANSNeuralStateUpdate(n_x_f=n_x, n_x_b=n_x_b, n_x_o=n_x, Binary_mat=torch.ones(n_x, n_x + n_x_b), n_u=n_u, n_feat=8, num_hidden_layers=1,
                              init_std=1e-2, neuron_list=[8, 8], act_fn=nn.Tanh(), include_bias=bias["f_b"], device=dev)
g_b = C.FANSNeuralStateUpdate(n_x_f=n_x, n_x_b=n_x_b, n_x_o=n_x_b, Binary_mat=torch.ones(n_x_b, n_x + n_x_b), n_u=n_u, n_feat=8, num_hidden_layers=1,
                              init_std=1e-2, neuron_list=[8, 8], act_fn=nn.Tanh(), include_bias=bias["g_b"], device=dev)
h_b = C.FANSNeuralOutputUpdate(n_x_f=n_x, n_x_b=n_x_b, n_x_o=n_y, Binary_mat=torch.ones(n_y, n_x + n_x_b), n_feat=8, num_hidden_layers=1,
                               init_std=1e-2, neuron_list=[8], act_fn=nn.Tanh(), include_bias=bias["h_b"], device=dev)
cnt = lambda m: int(sum(p.numel() for p in m.parameters()))
n_nn = cnt(f_b) + cnt(g_b) + cnt(h_b)
# physical part: theta_f (4) + x0 (2) ; augmented initial neural states x_b0 (3); learnable state normalisation mean/std (2 x (n_x+n_x_b) assumed)
extra = 4 + 2 + n_x_b
res = dict(f_b=cnt(f_b), g_b=cnt(g_b), h_b=cnt(h_b), nn_params=n_nn, physical_theta_f=4, x0_phys=2, x_b0=n_x_b,
           total_params_lower_bound=n_nn + extra, fp32_bytes_lower_bound=4 * (n_nn + extra))
x_f = torch.rand(1, n_x); x_b = torch.rand(1, n_x_b); u = torch.rand(1, n_u)
th = torch.tensor([0.2, 0.2, 0.2, 0.1])
def f_cascaded_tanks(x_step, u_step, theta_f):   # verbatim logic of user_config.f_cascaded_tanks (not imported: that module reads the test CSV at import)
    x_step = torch.clip(x_step, min=0)
    x_1 = x_step[:, 0:1]; x_2 = x_step[:, 1:2]
    t1, t3, t4, t5 = theta_f[0], theta_f[1], theta_f[2], theta_f[3]
    dx_1 = -t1 * torch.sqrt(x_1) + t4 * u_step
    x_1_overflow = x_1 + dx_1
    for b in range(x_step.shape[0]):
        if x_1_overflow[b] <= 10:
            dx_2 = t1 * torch.sqrt(x_1) - t3 * torch.sqrt(x_2)
        elif x_1_overflow[b] > 10:
            dx_2 = t1 * torch.sqrt(x_1) - t3 * torch.sqrt(x_2) + t5 * u_step
    return torch.cat((dx_1, dx_2), 1)


def step():
    with torch.no_grad():
        dx = f_cascaded_tanks(x_f, u, th) + f_b(x_f, x_b, u)
        db = g_b(x_f, x_b, u)
        y = x_f[:, 1:2] + h_b(x_f, x_b)
    return dx, db, y
for _ in range(200): step()
ts = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(3000): step()
    ts.append((time.perf_counter() - t0) / 3000 * 1e6)
res["torch_components_us_per_step_median"] = float(np.median(ts))
res["note"] = "instantiated with the Main.ipynb architecture (random weights: latency does not depend on values); components only, a lower bound on its per-sample latency (the repo simulator adds normalisation, projection and python loops). Measured on a heavily loaded machine."
json.dump(res, open(os.path.join(ROOT, "results", "ct_reference_edge.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
