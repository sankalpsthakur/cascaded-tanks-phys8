"""Append the round-2 test-record scorings of the final kernel to TEST-LOG.md (pure text processing).

The kernel writes its rows (numbered from 25, continuing the Cascaded Tanks rows) to TEST-LOG-CT2.md and mirrors them as test_log_ct2.json
({"text": ...}). This tool appends that section verbatim under its own header plus a provenance note. Refuses to run twice.
usage: python tools/ct2_ingest_testlog.py <kernel out dir> <kernel ref + version> <utc end of run> <freeze commit> <note>
"""
import json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kout, ref, utc, freeze_commit = sys.argv[1:5]
note_extra = sys.argv[5] if len(sys.argv) > 5 else ""
log_p = os.path.join(ROOT, "TEST-LOG.md")
local = open(log_p).read()
if "CT round 2: frozen" in local:
    sys.exit("already ingested")
text = json.load(open(os.path.join(kout, "test_log_ct2.json")))["text"]
rows = [l for l in text.splitlines() if re.match(r"\| \d+ \|", l)]
if not rows:
    sys.exit("the kernel copy has no rows")
tail = [
    "",
    f"Rows 25-{24 + len(rows)}: frozen-configuration run of `ct_round2` (`ct2/final2.py results/FREEZE_ct_round2.json --seeds 0..9`, freeze committed in {freeze_commit}), executed in the private Kaggle kernel `{ref}` (finished {utc} UTC).",
    "Every scoring of the test record by round-2 code is a row here, failures included; the `code @` column is the kernel's throw-away commit of the byte-identical sources (sha256 manifest, tree hashes of `ct/` and `ct2/` and the freeze blob checked against the control repo; see `logs/kaggle-output/ct-round2-final/out/manifest_verified.json`).",
    "Each seed was scored exactly once; nothing in these rows was used for selection." + (" " + note_extra if note_extra else ""),
]
with open(log_p, "a") as f:
    f.write("\n" + text.strip("\n") + "\n" + "\n".join(tail) + "\n")
print(f"appended {len(rows)} rows")
