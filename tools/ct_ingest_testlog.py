"""Append the test-record evaluations of the CT final kernel to the repo's TEST-LOG.md (pure text processing).

Checks that rows 1..14 of the kernel's copy are byte-identical to the local rows 1..14 (the kernel started from this file), then appends rows >= 15
verbatim plus a provenance note. Refuses to run twice (it stops when a row 15 already exists locally).
usage: python tools/ct_ingest_testlog.py <kernel out dir> <kernel ref + version> <utc end of run>
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kout, ref, utc = sys.argv[1], sys.argv[2], sys.argv[3]
local = open(os.path.join(ROOT, "TEST-LOG.md")).read().splitlines()
kern = open(os.path.join(kout, "TEST-LOG.md")).read().splitlines()
is_row = lambda l: re.match(r"\| \d+ \|", l) is not None
lrows = [l for l in local if is_row(l)]
krows = [l for l in kern if is_row(l)]
if len(lrows) != 14:
    sys.exit(f"local TEST-LOG has {len(lrows)} rows, expected 14 (already ingested?)")
if krows[:14] != lrows:
    sys.exit("rows 1..14 of the kernel copy differ from the local ones; not appending")
new = krows[14:]
if not new:
    sys.exit("the kernel log has no new rows")
note = [
    "",
    f"Rows 15-{14 + len(new)}: frozen-configuration run of `ct_phys8` (`ct/final.py results/FREEZE_ct_phys8.json --seeds 0..9`), executed in the private Kaggle kernel `{ref}` (finished {utc} UTC).",
    "The `code @` column shows the throw-away commit that the kernel made of the byte-identical sources (every file's sha256 and the git tree hash of `ct/`, 120f81f0, and of the freeze file, 55e4323c, matched the control repo at f0d4572; see `logs/kaggle-output/ct-final/out/manifest_verified.json` and the kernel log).",
    "Each seed was scored exactly once; nothing in these rows was used for selection. Further scoring of the test record (variants, ensembles, reversed split) would be logged below, labelled post-hoc.",
]
with open(os.path.join(ROOT, "TEST-LOG.md"), "a") as f:
    f.write("\n".join(new) + "\n" + "\n".join(note) + "\n")
print(f"appended {len(new)} rows")
