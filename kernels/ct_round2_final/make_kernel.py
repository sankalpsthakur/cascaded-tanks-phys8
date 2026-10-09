#!/usr/bin/env python3
"""Build kernels/ct_round2_final/script.py: kernel_main.py + an embedded tar.gz payload of the byte-identical committed sources and the freeze.

Reads files and git metadata only (no model code is run). Refuses to build when the committed sources, the freeze or the pre-registration differ from HEAD,
when ct/ no longer hashes to the round-1 tree, when the freeze says 'identical to ct_phys8' (then no test scoring is allowed), or when ct2/phys2.py or
ct2/select2.py differ from what the selection kernel verified.
usage (from the control repo root): python kernels/ct_round2_final/make_kernel.py
"""
import base64, hashlib, io, json, os, subprocess, sys, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KDIR = os.path.join(ROOT, "kernels", "ct_round2_final")
ROUND1_TREE_CT = "120f81f08ff5460b0a8d79758ebc28364ba9eca7"
FREEZE = "results/FREEZE_ct_round2.json"
PREREG = "PREREGISTRATION-CT-ROUND2.md"
TOOLS = ["tools/ct_aggregate.py", "tools/ct_data_check.py", "tools/ct2_edge_run.py", "tools/ct_edge_run.py"]
SEL_MANIFEST = os.path.join(ROOT, "logs", "kaggle-output", "ct-round2-select", "out", "manifest_verified.json")


def git(*a):
    return subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


dirty = git("status", "--porcelain", "--", "ct", "ct2", "results/FREEZE_*.json", PREREG, *TOOLS)
if dirty:
    sys.exit("REFUSING: committed sources differ from HEAD:\n" + dirty)
if git("rev-parse", "HEAD:ct") != ROUND1_TREE_CT:
    sys.exit("REFUSING: ct/ no longer hashes to the round-1 tree " + ROUND1_TREE_CT)
cfg = json.load(open(os.path.join(ROOT, FREEZE)))
if cfg.get("identical_to_ct_phys8"):
    sys.exit("REFUSING: the freeze is identical to ct_phys8; no new test scoring")
sel = json.load(open(SEL_MANIFEST))
for f in ("ct2/phys2.py", "ct2/select2.py"):
    if sel[f]["sha256"] != sha(os.path.join(ROOT, f)):
        sys.exit(f"REFUSING: {f} differs from the version the selection kernel verified")

files = sorted(f"ct/{n}" for n in os.listdir(os.path.join(ROOT, "ct")) if os.path.isfile(os.path.join(ROOT, "ct", n)))
files += sorted(f"ct2/{n}" for n in os.listdir(os.path.join(ROOT, "ct2")) if os.path.isfile(os.path.join(ROOT, "ct2", n)))
files += [FREEZE, ".gitignore", PREREG] + TOOLS
files += sorted(f"runs/{n}" for n in os.listdir(os.path.join(ROOT, "runs")) if n.startswith("ref_seed") and n.endswith(".npz"))
files += sorted(f"results/{n}" for n in os.listdir(os.path.join(ROOT, "results")) if n.startswith("ref_repro_seed") and n.endswith(".json"))
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
    repo_head=git("rev-parse", "HEAD"), git_tree_ct=git("rev-parse", "HEAD:ct"), git_tree_ct2=git("rev-parse", "HEAD:ct2"),
    git_blob_freeze=git("rev-parse", "HEAD:" + FREEZE), git_blob_prereg=git("rev-parse", "HEAD:" + PREREG),
    selection_kernel_sha256={f: sel[f]["sha256"] for f in ("ct2/phys2.py", "ct2/select2.py")},
    gate=cfg.get("gate"), structure=cfg["structure"]["id"], fit_variant=cfg["fit"]["variant"],
    data_zip_sha256=sha(zip_p), data_mat_sha256=sha(mat_p),
    manifest=manifest, executable=executable,
)
src = open(os.path.join(KDIR, "kernel_main.py")).read()
assert "__PAYLOAD_B64__" in src and "__EXPECT_JSON__" in src
src = src.replace("__PAYLOAD_B64__", payload).replace("__EXPECT_JSON__", json.dumps(expect, indent=1))
out = os.path.join(KDIR, "script.py")
open(out, "w").write(src)
print("wrote", out, os.path.getsize(out), "bytes;", len(files), "files;", "payload", len(payload), "b64 chars")
print("repo_head", expect["repo_head"], "tree ct", expect["git_tree_ct"], "tree ct2", expect["git_tree_ct2"], "freeze blob", expect["git_blob_freeze"])
