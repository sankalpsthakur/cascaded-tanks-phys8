#!/usr/bin/env python3
"""Build kernels/ct_final/script.py: kernel_main.py + an embedded tar.gz payload of the byte-identical frozen sources.

Reads files and git metadata only (no model code is run). Refuses to build when ct/ or the freeze json differ from HEAD.
usage (from the control repo root): python kernels/ct_final/make_kernel.py
"""
import base64, hashlib, io, json, os, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(ROOT, "kernels", "ct_final")
FREEZE = "results/FREEZE_ct_phys8.json"


def git(*a):
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


dirty = git("status", "--porcelain", "--", "ct", "results/FREEZE_*.json")
if dirty:
    sys.exit("REFUSING: frozen files differ from HEAD:\n" + dirty)

files = sorted(f"ct/{n}" for n in os.listdir(os.path.join(ROOT, "ct")) if os.path.isfile(os.path.join(ROOT, "ct", n)))
files += [FREEZE, ".gitignore", "TEST-LOG.md"]
files += sorted(f"runs/{n}" for n in os.listdir(os.path.join(ROOT, "runs")) if n.startswith("ref_seed") and n.endswith(".npz"))
files += sorted(f"results/{n}" for n in os.listdir(os.path.join(ROOT, "results")) if n.startswith("ref_repro_seed") and n.endswith(".json"))
files += ["tools/ct_aggregate.py", "tools/ct_edge_run.py", "tools/ct_data_check.py"]
assert len(files) == len(set(files))

buf = io.BytesIO()
manifest, executable = {}, []
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
    for rel in files:
        p = os.path.join(ROOT, rel)
        data = open(p, "rb").read()
        ti = tarfile.TarInfo(rel)
        ti.size = len(data)
        ti.mtime = 0
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mode = 0o755 if os.access(p, os.X_OK) else 0o644
        if ti.mode == 0o755:
            executable.append(rel)
        tf.addfile(ti, io.BytesIO(data))
        manifest[rel] = hashlib.sha256(data).hexdigest()
payload = base64.b64encode(buf.getvalue()).decode()

zip_p = os.path.join(ROOT, "data", "Cascaded_Tanks", "CascadedTanksFiles.zip")
mat_p = os.path.join(ROOT, "data", "Cascaded_Tanks", "CascadedTanksFiles", "dataBenchmark.mat")
expect = dict(
    repo_head=git("rev-parse", "HEAD"),
    git_tree_ct=git("rev-parse", "HEAD:ct"),
    git_blob_freeze=git("rev-parse", "HEAD:" + FREEZE),
    data_zip_sha256=sha(zip_p), data_mat_sha256=sha(mat_p),
    manifest=manifest, executable=executable,
)
src = open(os.path.join(KDIR, "kernel_main.py")).read()
assert "__PAYLOAD_B64__" in src and "__EXPECT_JSON__" in src
src = src.replace("__PAYLOAD_B64__", payload).replace("__EXPECT_JSON__", json.dumps(expect, indent=1))
out = os.path.join(KDIR, "script.py")
open(out, "w").write(src)
print("wrote", out, os.path.getsize(out), "bytes;", len(files), "files;", "payload", len(payload), "b64 chars")
print("repo_head", expect["repo_head"], "tree ct", expect["git_tree_ct"], "freeze blob", expect["git_blob_freeze"])
