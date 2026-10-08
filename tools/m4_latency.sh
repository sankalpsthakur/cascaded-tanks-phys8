#!/bin/bash
# The ONE local-compute exception (owner rule): edge latency on this M4 for a model under 50 MB, 1 thread, batch 1,
# through ../bin/guard.sh -t 60 -m 512, and only while `uptime` shows a 1-minute load average under 4.
#   tools/m4_latency.sh ct   <dir holding ct_step.c and bench_main.c>   -> results/m4_latency_ct.json
#   tools/m4_latency.sh onnx <file.onnx> [file.onnx ...]                  -> results/m4_latency_pg.json
cd "$(dirname "$0")/.." || exit 1
GUARD=../bin/guard.sh
LOAD=$(uptime | sed 's/.*load averages*: *//' | awk '{print $1}' | tr -d ',')
STAMP=$(date -u +%Y-%m-%dT%H:%MZ)
if ! awk -v l="$LOAD" 'BEGIN{exit !(l < 4)}'; then
  echo "latency deferred (machine busy): 1-min load average $LOAD at $STAMP"
  exit 0
fi
case "$1" in
  ct)
    D=$2
    $GUARD -t 60 -m 512 -- clang -O2 -std=c99 -DCT_MAIN -o "$D/ct_main_m4" "$D/ct_step.c" -lm || exit 2
    $GUARD -t 60 -m 512 -- clang -O2 -std=c99 -I "$D" -o "$D/ct_bench_m4" "$D/bench_main.c" -lm || exit 2
    BENCH=$( for i in 1 2 3 4 5; do $GUARD -t 60 -m 512 -- "$D/ct_bench_m4" 20000 | awk '{print $1}'; done | tr '\n' ' ')
    BENCH2=$( for i in 1 2 3; do $GUARD -t 60 -m 512 -- "$D/ct_main_m4" --bench | awk '{print $1}'; done | tr '\n' ' ')
    printf '{"utc":"%s","load_1min":%s,"chip":"%s","step_only_ns_runs":"%s","bench_incl_sin_ns_runs":"%s","flags":"clang -O2 -std=c99","threads":1}\n' \
      "$STAMP" "$LOAD" "$(sysctl -n machdep.cpu.brand_string)" "$BENCH" "$BENCH2" | tee results/m4_latency_ct.json;;
  onnx)
    shift
    $GUARD -t 60 -m 512 -- .venv-pcgym/bin/python tools/m4_latency_onnx.py "$@" | tee results/m4_latency_pg.json;;
esac
