"""Official Cascaded Tanks protocol + test-access guard.

Design rule: the official TEST record is only reachable through `test_eval()` /
`load_test_logged()`. Every call appends a row to TEST-LOG.md (including failed
runs and reproduction runs of other people's code). Selection code (CV, tuning)
must only ever call `load_train()`.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TEST_LOG = ROOT / "TEST-LOG.md"
GDRIVE_URL = "https://drive.google.com/file/d/1HnQf_gu0g_UlggoBqy2s34l9YJiFdN01/view"  # loader's 2nd official URL
N_INIT = 50  # nonlinear_benchmarks sets test.state_initialization_window_length = 50
Y_CLIP = 10.0  # both records saturate at exactly 10.0 V


def _load_both():
    import nonlinear_benchmarks as nb

    DATA_DIR.mkdir(exist_ok=True)
    return nb.Cascaded_Tanks(dir_placement=str(DATA_DIR), url=GDRIVE_URL)


def load_train():
    """Estimation record only (u, y as float64 1-D arrays, Ts)."""
    train, _ = _load_both()
    return np.asarray(train.u, float).ravel(), np.asarray(train.y, float).ravel(), float(train.sampling_time)


def git_head() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", "ct", "results/FREEZE_*.json"], text=True, stderr=subprocess.DEVNULL).strip())
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "nogit"


def config_id(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


_HEADER = (
    "# TEST-LOG\n\n"
    "Every evaluation that touches the official Cascaded Tanks TEST record is appended here, including\n"
    "reproduction runs of third-party code and failed runs. Nothing in this table was used for selection\n"
    "unless the `purpose` column says so explicitly (it should never).\n\n"
    "RMSE column = official `nonlinear_benchmarks` metric: free-run simulation, first 50 test samples are the\n"
    "state-initialisation window and excluded (n_init=50), volts.\n\n"
    "| # | UTC | tag | config | seed | RMSE n_init=50 | extra | purpose | code @ |\n"
    "|---|-----|-----|--------|------|----------------|-------|---------|--------|\n"
)


def _append_log(row: dict) -> int:
    import fcntl

    with open(ROOT / ".testlog.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not TEST_LOG.exists():
            TEST_LOG.write_text(_HEADER)
        n = sum(1 for line in TEST_LOG.read_text().splitlines() if line.startswith("| ") and line[2:3].isdigit()) + 1
        line = "| {n} | {utc} | {tag} | {cfg} | {seed} | {rmse} | {extra} | {purpose} | {code} |\n".format(
            n=n,
            utc=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            tag=row.get("tag", ""),
            cfg=row.get("config", ""),
            seed=row.get("seed", ""),
            rmse=row.get("rmse", ""),
            extra=str(row.get("extra", "")).replace("|", "/"),
            purpose=row.get("purpose", ""),
            code=git_head(),
        )
        with open(TEST_LOG, "a") as f:
            f.write(line)
        return n


def official_rmse(y_true_full: np.ndarray, y_model_full: np.ndarray) -> float:
    """Official metric via the package function, on full-length (1024) arrays."""
    from nonlinear_benchmarks.error_metrics import RMSE

    y_true_full = np.asarray(y_true_full, float).reshape(-1)
    y_model_full = np.asarray(y_model_full, float).reshape(-1)
    assert y_true_full.shape == y_model_full.shape == (1024,), (y_true_full.shape, y_model_full.shape)
    return float(RMSE(y_true_full, y_model_full, n_init=N_INIT))


def test_eval(simulate_fn, tag: str, config: dict, seed, purpose: str, extra: str = "", log: bool = True):
    """The ONLY sanctioned way to score on the official test record.

    simulate_fn(u_test, y_init_window) -> y_model (len 1024). It receives the whole test input
    and only the first N_INIT outputs (the official state-initialisation window).
    """
    _, test = _load_both()
    u = np.asarray(test.u, float).ravel()
    y = np.asarray(test.y, float).ravel()
    assert test.state_initialization_window_length == N_INIT
    y_model = np.asarray(simulate_fn(u, y[:N_INIT].copy()), float).reshape(-1)
    rmse = official_rmse(y, y_model)
    if log:
        _append_log(dict(tag=tag, config=config_id(config), seed=seed, rmse=f"{rmse:.4f}", extra=extra, purpose=purpose))
    return rmse, y_model


def log_external(tag: str, config: dict, seed, rmse, extra: str, purpose: str):
    """Log a test-set evaluation that was computed outside test_eval (e.g. third-party code)."""
    return _append_log(dict(tag=tag, config=config_id(config), seed=seed, rmse=rmse, extra=extra, purpose=purpose))
