#!/usr/bin/env python3
"""Build kernels/ct_round2_select/script.py: kernel_main.py + an embedded tar.gz payload of the byte-identical committed sources.

Reads files and git metadata only (no model code is run). Refuses to build when ct/, ct2/, the freeze files or the pre-registration differ from HEAD,
or when ct/ no longer hashes to the round-1 tree (120f81f0...).
usage (from the control repo root): python kernels/ct_round2_select/make_kernel.py
"""
import base64, hashlib, io, json, os, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(ROOT, "kernels", "ct_round2_select")
ROUND1_TREE_CT = "120f81f08ff5460b0a8d79758ebc28364ba9eca7"
PREREG = "PREREGISTRATION-CT-ROUND2.md"
EXTRA = ["results/FREEZE_ct_phys8.json", "results/ct_final_ct_phys8.json", ".gitignore", PREREG, "tools/ct_data_check.py"]


def git(*a):
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


dirty = git("status", "--porcelain", "--", "ct", "ct2", "results/FREEZE_*.json", "results/ct_final_ct_phys8.json", PREREG, "tools/ct_data_check.py")
if dirty:
    sys.exit("REFUSING: committed sources differ from HEAD:\n" + dirty)
if git("rev-parse", "HEAD:ct") != ROUND1_TREE_CT:
    sys.exit("REFUSING: ct/ no longer hashes to the round-1 tree " + ROUND1_TREE_CT)

files = sorted(f"ct/{n}" for n in os.listdir(os.path.join(ROOT, "ct")) if os.path.isfile(os.path.join(ROOT, "ct", n)))
files += sorted(f"ct2/{n}" for n in os.listdir(os.path.join(ROOT, "ct2")) if os.path.isfile(os.path.join(ROOT, "ct2", n)))
files += EXTRA
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
    git_tree_ct=git("rev-parse", "HEAD:ct"), git_tree_ct2=git("rev-parse", "HEAD:ct2"),
    git_blob_prereg=git("rev-parse", "HEAD:" + PREREG),
    data_zip_sha256=sha(zip_p), data_mat_sha256=sha(mat_p),
    manifest=manifest, executable=executable,
)
src = open(os.path.join(KDIR, "kernel_main.py")).read()
assert "__PAYLOAD_B64__" in src and "__EXPECT_JSON__" in src
src = src.replace("__PAYLOAD_B64__", payload).replace("__EXPECT_JSON__", json.dumps(expect, indent=1))
out = os.path.join(KDIR, "script.py")
open(out, "w").write(src)
print("wrote", out, os.path.getsize(out), "bytes;", len(files), "files;", "payload", len(payload), "b64 chars")
print("repo_head", expect["repo_head"], "tree ct", expect["git_tree_ct"], "tree ct2", expect["git_tree_ct2"], "prereg blob", expect["git_blob_prereg"])
