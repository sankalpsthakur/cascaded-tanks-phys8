"""Markdown tables from the round-2 selection kernel's selection.json (pure text processing; reads no data).
usage: python tools/ct2_tables.py <selection.json>"""
import json, sys

s = json.load(open(sys.argv[1]))
print("reproduction gate:", json.dumps(s.get("reproduction_gate")))
if "abort" in s:
    print("ABORT:", s["abort"])


def fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


st1 = s.get("stage1")
if st1:
    r = st1["rule"]
    print("\n| id | structure | numbers | block RMSE (4 blocks) | pooled CV | SE | level RMSE | eligible | secs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for cid, row in st1["rows"].items():
        if row["complete"]:
            flag = ("chosen" if cid == st1["chosen"] else ("yes" if cid in r["eligible"] else "no"))
            print(f"| {cid} | {row['label']} | {row['n_numbers']} | {', '.join(fmt(f) for f in row['folds'])} | {row['cv']:.4f} | {row['se']:.4f} | {fmt(row['level_rmse'])} | {flag} | {row['secs']:.0f} |")
        else:
            print(f"| {cid} | {row['label']} | {row['n_numbers']} | incomplete | - | - | - | excluded ({row.get('error')}) | {row['secs']:.0f} |")
    print(f"\nrule 1: best {r['best']} CV {r['best_cv']:.4f}, SE {r['se_best']:.4f}, threshold {r['threshold']:.4f}; eligible {r['eligible']}; fewest numbers {r['n_numbers_min']}: {r['tied_on_size']}; level RMSE ties {r['tied_on_level']}; chosen {st1['chosen']}")
st2 = s.get("stage2")
if st2:
    r = st2["rule"]
    print(f"\nstage 2 on {st2['structure']}\n\n| fit | numbers | block RMSE | pooled CV | SE | level RMSE | eligible |")
    print("|---|---|---|---|---|---|---|")
    for vid, row in st2["rows"].items():
        if row["complete"]:
            flag = ("chosen" if vid == st2["chosen"] else ("yes" if vid in r["eligible"] else "no"))
            print(f"| {vid} | {row['n_numbers']} | {', '.join(fmt(f) for f in row['folds'])} | {row['cv']:.4f} | {row['se']:.4f} | {fmt(row['level_rmse'])} | {flag} |")
        else:
            print(f"| {vid} | {row['n_numbers']} | incomplete ({row.get('error')}) | - | - | - | excluded |")
    print(f"\nrule 2: best {r['best']} CV {r['best_cv']:.4f}, SE {r['se_best']:.4f}, threshold {r['threshold']:.4f}; eligible {r['eligible']}; level ties {r['tied_on_level']}; chosen {st2['chosen']}")
fin = s.get("final")
if fin:
    d = fin["dry_run"]
    print(f"\nfinal: {fin['structure']} + {fin['variant']} ({fin['variant_spec']['label']}); dry run train RMSE {d['train_rmse']:.5f} (objective {d['objective']:.6f}, {d['secs']:.0f} s)")
    print("identity:", json.dumps({k: v for k, v in fin["identical_to_ct_phys8"].items() if k != "rel_diff"}))
    print("theta:", json.dumps({k: round(v, 5) for k, v in d["theta"].items()}))
print("elapsed s:", s.get("elapsed_s"))
