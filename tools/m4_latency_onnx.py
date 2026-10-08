"""ONNX Runtime latency of tiny policy graphs on this machine: 1 intra-op thread, batch 1, median/min over 5 x 20000 runs.
Invoked only by tools/m4_latency.sh (load-average gate + guard.sh)."""
import json, os, sys, time, datetime
import numpy as np
import onnxruntime as ort

res = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "ort": ort.__version__, "threads": 1, "load_1min": os.getloadavg()[0], "models": {}}
for path in sys.argv[1:]:
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    n_in = s.get_inputs()[0].shape[1]
    x = np.random.default_rng(0).uniform(-1, 1, (1, n_in)).astype(np.float32)
    for _ in range(500): s.run(None, {"x": x})
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20000): s.run(None, {"x": x})
        ts.append((time.perf_counter() - t0) / 20000 * 1e6)
    res["models"][os.path.basename(path)] = dict(bytes=os.path.getsize(path), us_median=float(np.median(ts)), us_min=float(np.min(ts)))
print(json.dumps(res))
