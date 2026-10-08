"""Print a Kaggle kernel log (the JSON list written by `kaggle kernels output`) as plain text.
usage: python tools/kaggle_log.py <kernel>.log [--grep PATTERN] [--tail N]"""
import json, re, sys

path = sys.argv[1]
items = json.loads(open(path, encoding="utf-8", errors="replace").read())
text = "".join(i.get("data", "") for i in items)
lines = text.splitlines()
if "--grep" in sys.argv:
    pat = re.compile(sys.argv[sys.argv.index("--grep") + 1])
    lines = [l for l in lines if pat.search(l)]
if "--tail" in sys.argv:
    lines = lines[-int(sys.argv[sys.argv.index("--tail") + 1]):]
print("\n".join(lines))
